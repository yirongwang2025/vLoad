import asyncio
import json
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)
import bisect
import subprocess
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Deque, Set, Optional, Dict, Any, Callable, Tuple, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routers.ws import manager
from routers.sessions import session_start, session_stop
from schemas.requests import ConnectPayload, SessionStartPayload
from app_state import AppState
from fastapi import Depends
from deps import get_state

# Reuse your BLE client
from modules.movesense_gatt import MovesenseGATTClient
from modules import db
from modules.jump_detector import JumpDetectorRealtime
from modules.video_backend import get_video_backend, start_video_collector_subprocess
from modules.config import get_config, get_jump_detection_defaults

# ----------------------------
# Pose auto-run (best-effort)
# ----------------------------


async def _run_pose_for_jump_best_effort(event_id: int, max_fps: Optional[float] = None) -> Dict[str, Any]:
	"""
	Run pose on a jump clip and persist pose_* columns. Designed for background use.
	If pose deps are missing, store a useful error in pose_meta and return {ok: False}.
	"""
	ev = int(event_id)
	try:
		row = await db.get_jump_with_imu(ev)
		if not row:
			return {"ok": False, "error": "jump not found"}
		video_path = row.get("video_path")
		if not video_path:
			return {"ok": False, "error": "video_path missing"}
		p = Path(str(video_path))
		if not p.exists():
			return {"ok": False, "error": f"clip not found: {video_path}"}

		t0v = row.get("t_takeoff_video_t")
		t1v = row.get("t_landing_video_t")
		if not (isinstance(t0v, (int, float)) and isinstance(t1v, (int, float)) and float(t1v) > float(t0v)):
			return {"ok": False, "error": "marks not set"}

		try:
			max_fps = float(max_fps) if max_fps is not None else float(CFG.pose.max_fps)
		except Exception:
			max_fps = float(CFG.pose.max_fps)
		max_fps = max(1.0, min(30.0, float(max_fps)))

		# Lazy imports so the server can run without pose deps installed.
		try:
			import cv2  # type: ignore
		except Exception as e:
			meta = {
				"error": "opencv import failed",
				"detail": repr(e),
				"install": "pip install -r requirements_pose.txt",
			}
			await db.update_jump_pose_metrics(event_id=ev, pose_meta=meta)
			return {"ok": False, "error": "opencv import failed", "pose_meta": meta}

		try:
			from modules.pose.mediapipe_provider import MediaPipePoseProvider
			from modules.pose.pose_metrics import estimate_revolutions_from_shoulders, height_from_flight_time, summarize_pose_run
		except Exception as e:
			meta = {
				"error": "mediapipe import failed",
				"detail": repr(e),
				"install": "pip install -r requirements_pose.txt",
			}
			await db.update_jump_pose_metrics(event_id=ev, pose_meta=meta)
			return {"ok": False, "error": "mediapipe import failed", "pose_meta": meta}

		cap = cv2.VideoCapture(str(p))
		if not cap.isOpened():
			meta = {"error": "failed to open clip", "clip_path": str(video_path)}
			await db.update_jump_pose_metrics(event_id=ev, pose_meta=meta)
			return {"ok": False, "error": "failed to open clip", "pose_meta": meta}
		fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
		if not (fps > 0):
			fps = 30.0
		stride = int(max(1, round(float(fps) / float(max_fps))))

		start_frame = int(max(0, round(float(t0v) * float(fps))))
		end_frame = int(max(start_frame + 1, round(float(t1v) * float(fps))))
		cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

		provider = MediaPipePoseProvider(
			model_complexity=int(CFG.pose.model_complexity),
			min_detection_confidence=float(CFG.pose.min_detection_confidence),
			min_tracking_confidence=float(CFG.pose.min_tracking_confidence),
		)
		frames = []
		i = start_frame
		try:
			while i <= end_frame:
				ok, frame_bgr = cap.read()
				if not ok or frame_bgr is None:
					break
				if (i - start_frame) % stride != 0:
					i += 1
					continue
				frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
				t_video = float(i) / float(fps)
				frames.append(provider.infer_rgb(frame_rgb, t_video=t_video))
				i += 1
		finally:
			try:
				provider.close()
			except Exception:
				pass
			try:
				cap.release()
			except Exception:
				pass

		flight_time_pose = float(t1v) - float(t0v)
		height_pose = height_from_flight_time(flight_time_pose)
		revs_pose = estimate_revolutions_from_shoulders(frames)
		pose_meta = summarize_pose_run(frames)
		pose_meta.update(
			{
				"clip_path": str(video_path),
				"fps": float(fps),
				"stride": int(stride),
				"t_takeoff_video_t": float(t0v),
				"t_landing_video_t": float(t1v),
				"rotation_method": "shoulder_line_angle_2d_proxy",
				"note": "Prototype: 2D proxy rotation; height from projectile model using marked flight time.",
			}
		)

		await db.update_jump_pose_metrics(
			event_id=ev,
			flight_time_pose=float(flight_time_pose),
			height_pose=float(height_pose),
			revolutions_pose=(float(revs_pose) if revs_pose is not None else None),
			pose_meta=pose_meta,
		)
		return {
			"ok": True,
			"event_id": ev,
			"flight_time_pose": float(flight_time_pose),
			"height_pose": float(height_pose),
			"revolutions_pose": (float(revs_pose) if revs_pose is not None else None),
			"pose_meta": pose_meta,
		}
	except Exception as e:
		meta = {"error": "pose run exception", "detail": repr(e)}
		try:
			await db.update_jump_pose_metrics(event_id=ev, pose_meta=meta)
		except Exception:
			pass
		return {"ok": False, "error": "pose run exception", "pose_meta": meta}


def _maybe_schedule_pose_for_jump(event_id: int) -> None:
	"""
	Fire-and-forget scheduling. Never raises.
	"""
	ev = int(event_id)
	if _app_state is None:
		return
	inflight = _app_state.pose_jobs_inflight
	if ev in inflight:
		return
	inflight.add(ev)

	async def _runner():
		try:
			await _run_pose_for_jump_best_effort(ev, max_fps=float(CFG.pose.max_fps))
		finally:
			try:
				inflight.discard(ev)
			except Exception:
				pass

	try:
		asyncio.create_task(_runner())
	except Exception:
		try:
			inflight.discard(ev)
		except Exception:
			pass

# UI directory path
UI_DIR = Path(__file__).parent / "UI"


def load_html_template(filename: str) -> str:
	"""
	Load an HTML template file from the UI directory.
	
	Args:
		filename: Name of the HTML file (e.g., 'index.html')
		
	Returns:
		The HTML content as a string
		
	Raises:
		FileNotFoundError: If the file doesn't exist
	"""
	file_path = UI_DIR / filename
	if not file_path.exists():
		raise FileNotFoundError(f"UI template not found: {file_path}")
	with open(file_path, 'r', encoding='utf-8') as f:
		return f.read()


# Lazy HTML cache: no template read at import; load on first request (B.1 Option B)
_html_cache: Dict[str, str] = {}


def get_page_html(filename: str) -> str:
	"""Load HTML template on first request and cache. Server can start without UI/."""
	if filename not in _html_cache:
		_html_cache[filename] = load_html_template(filename)
	return _html_cache[filename]


CFG = get_config()
DEVICE = (CFG.movesense.default_device or "").strip()  # UI default only
MODE = (CFG.movesense.default_mode or "IMU9").strip()  # "IMU6" or "IMU9"
RATE = int(CFG.movesense.default_rate or 104)
APP_NAME = "P2Skating"
APP_VERSION_FILE = Path(__file__).resolve().parent / "VERSION"


def _read_app_version() -> str:
	"""Return software version from VERSION file (fallback: 0.1.0)."""
	try:
		if APP_VERSION_FILE.exists():
			v = APP_VERSION_FILE.read_text(encoding="utf-8").strip()
			if v:
				return v
	except Exception:
		pass
	return "0.1.0"

# IMU collector / UDP (config only; runtime state lives in AppState)
IMU_UDP_HOST = (CFG.imu_udp.host or "127.0.0.1").strip()
IMU_UDP_PORT = int(CFG.imu_udp.port or 9999)

# Jump clip worker (config only; process ref lives in AppState)
JUMP_CLIP_JOBS_DIR = CFG.jobs.jump_clip_jobs_dir

# Jump detection defaults come from config.json (modules/config.py)
JUMP_CONFIG_DEFAULTS: Dict[str, float] = get_jump_detection_defaults(CFG)

# Constants for buffer sizing (used in lifespan / workers)
IMU_HISTORY_MAX_SECONDS: float = float(CFG.buffers.imu_history_seconds)
FRAME_HISTORY_MAX_SECONDS: float = float(CFG.buffers.frame_history_seconds)

# 3.1: single app state instance; set in lifespan, used by helpers that run outside request context
_app_state: Optional[AppState] = None


async def _do_connect_imu_for_skater(st: AppState, skater_id: int) -> bool:
	"""Start IMU collector for the given skater. Returns True if started, False on error."""
	try:
		skater_devices = await db.get_skater_devices(int(skater_id))
		if not skater_devices:
			return False
		waist_device = next((d for d in skater_devices if d.get("placement") == "waist"), None)
		selected_device = waist_device or skater_devices[0]
		device = selected_device.get("mac_address")
		if not device:
			return False
		resolved_mac = await db.resolve_device_identifier(device)
		if resolved_mac:
			device = resolved_mac
		if st.imu_proc is not None:
			try:
				st.imu_proc.terminate()
			except Exception:
				pass
			try:
				st.imu_proc.wait(timeout=float(CFG.runtime.subprocess_wait_timeout_seconds))
			except Exception:
				pass
			st.imu_proc = None
		cmd = [
			sys.executable,
			"-m",
			"modules.imu_collector",
			"--device",
			device,
			"--mode",
			MODE,
			"--rate",
			str(RATE),
			"--udp-host",
			IMU_UDP_HOST,
			"--udp-port",
			str(IMU_UDP_PORT),
		]
		st.imu_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		try:
			st.dbg["collector_running"] = True
			st.dbg["collector_pid"] = int(st.imu_proc.pid) if st.imu_proc.pid else None
		except Exception:
			pass
		if st.log_to_clients:
			st.log_to_clients(f"[AutoConnect] Started IMU collector for skater {skater_id} (device={device})")
		return True
	except Exception:
		return False


async def _auto_connect_loop(st: AppState) -> None:
	"""
	On startup: if a default skater exists, auto-connect video and IMU independently.
	- Video: connect once (even if IMU is unavailable).
	- IMU: if default skater has a device, try to connect; retry every imu_retry_interval_seconds until connected.
	"""
	retry_sec = max(1.0, float(CFG.auto_connect.imu_retry_interval_seconds))
	skater = await db.get_default_skater()
	if not skater or not skater.get("id"):
		return
	skater_id = int(skater["id"])

	# Connect video independently (do not block on IMU)
	try:
		st.video.start()
		if st.log_to_clients:
			st.log_to_clients("[AutoConnect] Video started")
	except Exception as e:
		if st.log_to_clients:
			st.log_to_clients(f"[AutoConnect] Video start failed: {e!r}")

	# IMU: only if default skater has a device
	devices = await db.get_skater_devices(skater_id)
	if not devices:
		if st.log_to_clients:
			st.log_to_clients("[AutoConnect] Default skater has no IMU device; video only")
		return

	while True:
		try:
			ok = await _do_connect_imu_for_skater(st, skater_id)
			if not ok:
				if st.log_to_clients:
					st.log_to_clients(f"[AutoConnect] Failed to start IMU collector; retrying in {retry_sec:.0f}s")
				await asyncio.sleep(retry_sec)
				continue
			await asyncio.sleep(retry_sec)
			last_t = st.dbg.get("last_imu_t")
			hist_len = len(st.imu_history) if st.imu_history else 0
			if (last_t is not None and last_t > 0) or hist_len > 0:
				st.jump_detection_enabled = bool(CFG.auto_connect.jump_detection_enabled)
				# Auto-connect can enable detection without hitting /detection/start.
				# Ensure we also have an active recording session for clip generation.
				if st.jump_detection_enabled and st.session_id is None and st.detection_session_id is None:
					try:
						sid = time.strftime("%Y%m%d_%H%M%S") + "_detect"
						resp = await session_start(SessionStartPayload(session_id=sid), state=st)
						resolved_sid = (
							(resp.get("session_id") if isinstance(resp, dict) else None)
							or st.session_id
							or sid
						)
						st.detection_session_id = str(resolved_sid)
					except Exception as e:
						if st.log_to_clients:
							st.log_to_clients(f"[AutoConnect] Could not auto-start detection session: {e!r}")
				if st.log_to_clients:
					st.log_to_clients(
						f"[AutoConnect] IMU connected successfully (detection={'on' if st.jump_detection_enabled else 'off'})"
					)
				break
			if st.log_to_clients:
				st.log_to_clients(f"[AutoConnect] No IMU data yet; retrying in {retry_sec:.0f}s...")
			if st.imu_proc is not None:
				try:
					st.imu_proc.terminate()
				except Exception:
					pass
				try:
					st.imu_proc.wait(timeout=float(CFG.runtime.subprocess_wait_timeout_seconds))
				except Exception:
					pass
				st.imu_proc = None
			await asyncio.sleep(retry_sec)
		except asyncio.CancelledError:
			break
		except Exception as e:
			if st.log_to_clients:
				st.log_to_clients(f"[AutoConnect] Error: {e!r}")
			await asyncio.sleep(retry_sec)


def _jump_log_filter(message: str) -> None:
	"""
	Filter JumpDetector log lines before sending them to clients.

	- Always forward Phase 1/2 diagnostics.
	- Only forward [Jump] lines once detection has been manually enabled.
	"""
	enabled = _app_state.jump_detection_enabled if _app_state else False
	if "[Jump]" in message and not enabled:
		return
	_log_to_clients(f"[JumpDetector] {message}")


@asynccontextmanager
async def lifespan(app: FastAPI):
	global _app_state
	try:
		# Initialise database (if configured).
		try:
			await db.init_db()
		except Exception as e:
			# DB is optional; log to clients and continue without persistence.
			logger.error("[DB] init_db failed: %s", e)
			# We deliberately don't call _log_to_clients here because WS manager
			# may not be ready yet.

		# 3.1: Create single AppState and attach to app + module ref for helpers outside request context.
		app_state = AppState()
		app_state.manager = manager
		app_state.get_page_html = get_page_html
		app_state.UI_DIR = UI_DIR
		app_state.video = get_video_backend(CFG)
		app_state.cfg = CFG
		app_state.active_mode = MODE
		app_state.active_rate = RATE
		app_state.jump_config = dict(JUMP_CONFIG_DEFAULTS)
		app_state.jump_detection_enabled = False
		app_state.session_base_dir = _session_base_dir
		app_state.log_to_clients = _log_to_clients
		app_state.enqueue_jump_clip_job = _enqueue_jump_clip_job
		app_state.run_pose_for_jump_best_effort = _run_pose_for_jump_best_effort
		app_state.maybe_schedule_pose_for_jump = _maybe_schedule_pose_for_jump
		app_state.count_clip_jobs_pending = _count_clip_jobs_pending
		app_state.count_clip_jobs_done = _count_clip_jobs_done
		app_state.count_clip_jobs_failed = _count_clip_jobs_failed
		app_state.read_last_clip_job_error = _read_last_clip_job_error
		app_state.session_lock = asyncio.Lock()
		# Buffers and queues (created here, consumed by workers and UDP)
		app_state.jump_sample_queue = asyncio.Queue(maxsize=int(CFG.buffers.jump_sample_queue_maxsize))
		app_state.imu_history = deque()
		app_state.frame_history = deque()
		app_state.jump_events = deque(maxlen=int(CFG.buffers.jump_events_maxlen))
		app_state.next_event_id = 1
		app_state.imu_rx_window = deque()
		app_state.imu_rx_window_pkts = deque()
		app.state.state = app_state
		_app_state = app_state

		# Start decoupled jump‑analysis worker and frame sync task.
		app_state.jump_worker_task = asyncio.create_task(_jump_worker_loop(app_state))
		app_state.frame_sync_task = asyncio.create_task(_frame_sync_loop(app_state))

		# Start UDP receiver for IMU packets from collector process.
		loop = asyncio.get_running_loop()

		class _ImuUdpProtocol(asyncio.DatagramProtocol):
			def __init__(self, st: AppState) -> None:
				self.st = st

			def datagram_received(self, data: bytes, addr):  # type: ignore[override]
				st = self.st
				# Parse JSON (best-effort)
				try:
					msg = json.loads(data.decode("utf-8"))
				except Exception:
					return

				# Collector log passthrough
				if isinstance(msg, dict) and msg.get("type") == "log":
					txt = msg.get("msg")
					if isinstance(txt, str) and st.log_to_clients:
						st.log_to_clients(txt)
					return

				# Collector telemetry passthrough
				if isinstance(msg, dict) and msg.get("type") == "collector_stat":
					try:
						if "disconnects" in msg:
							st.dbg["collector_disconnects"] = int(msg.get("disconnects") or 0)
						if "last_disconnect_t" in msg:
							st.dbg["collector_last_disconnect_t"] = msg.get("last_disconnect_t")
						if "last_error" in msg:
							st.dbg["collector_last_error"] = msg.get("last_error")
						if "rx_samples_5s" in msg:
							st.dbg["collector_rx_samples_5s"] = msg.get("rx_samples_5s")
						if "rx_packets_5s" in msg:
							st.dbg["collector_rx_packets_5s"] = msg.get("rx_packets_5s")
						if "rx_rate_hz_5s" in msg:
							st.dbg["collector_rx_rate_hz_5s"] = msg.get("rx_rate_hz_5s")
						if "notify_stale_s" in msg:
							st.dbg["collector_notify_stale_s"] = msg.get("notify_stale_s")
					except Exception:
						pass
					return

				# IMU packet from collector
				if not isinstance(msg, dict) or msg.get("type") != "imu":
					return

				try:
					st.dbg["collector_last_pkt_t"] = float(time.time())
				except Exception:
					pass

				# Collector-provided clock calibration (device timestamp -> epoch seconds)
				try:
					if isinstance(msg.get("imu_clock_offset_s"), (int, float)):
						st.dbg["imu_clock_offset_s"] = float(msg.get("imu_clock_offset_s"))
				except Exception:
					pass
				try:
					if isinstance(msg.get("imu_clock_offset_fixed"), bool):
						st.dbg["imu_clock_offset_fixed"] = bool(msg.get("imu_clock_offset_fixed"))
					if isinstance(msg.get("imu_clock_offset_calib_n"), (int, float)):
						st.dbg["imu_clock_offset_calib_n"] = int(msg.get("imu_clock_offset_calib_n"))
				except Exception:
					pass

				# Update rate/mode if present (collector is authoritative once running)
				try:
					if isinstance(msg.get("rate"), (int, float)):
						st.active_rate = int(msg.get("rate"))
					if isinstance(msg.get("mode"), str):
						st.active_mode = str(msg.get("mode"))
				except Exception:
					pass

				samples = msg.get("samples") or []
				if not isinstance(samples, list) or not samples:
					return

				# Server-side rolling 5s receive rate (independent of browser)
				try:
					now_m = time.monotonic()
					st.imu_rx_window.append((now_m, int(len(samples))))
					st.imu_rx_window_pkts.append((now_m, 1))
					cut = now_m - float(CFG.runtime.imu_rx_rate_window_seconds)
					while st.imu_rx_window and st.imu_rx_window[0][0] < cut:
						st.imu_rx_window.popleft()
					while st.imu_rx_window_pkts and st.imu_rx_window_pkts[0][0] < cut:
						st.imu_rx_window_pkts.popleft()
					s5 = sum(n for _, n in st.imu_rx_window)
					p5 = len(st.imu_rx_window_pkts)
					st.dbg["imu_samples_5s"] = int(s5)
					st.dbg["imu_packets_5s"] = int(p5)
					st.dbg["imu_rate_hz_5s"] = float(s5) / float(CFG.runtime.imu_rx_rate_window_seconds)
				except Exception:
					pass

				# Update debug counters based on collector payload.
				try:
					st.dbg["imu_packets"] = int(st.dbg.get("imu_packets", 0)) + 1
					st.dbg["imu_samples"] = int(st.dbg.get("imu_samples", 0)) + int(len(samples))
				except Exception:
					pass

				# Push samples into jump queue + history + recording (best effort).
				_last_t_i: Optional[float] = None
				for s in samples:
					if not isinstance(s, dict):
						continue
					try:
						t_i = float(s.get("t"))
					except Exception:
						continue
					_last_t_i = t_i

					# Jump worker queue
					if st.jump_sample_queue is not None:
						try:
							st.jump_sample_queue.put_nowait(
								{
									"t": t_i,
									"acc": s.get("acc", []),
									"gyro": s.get("gyro", []),
									"mag": s.get("mag", []),
									"imu_timestamp": s.get("imu_timestamp"),
									"imu_sample_index": s.get("imu_sample_index"),
								}
							)
							st.dbg["jump_queue_put_ok"] = int(st.dbg.get("jump_queue_put_ok", 0)) + 1
						except Exception:
							try:
								st.dbg["jump_queue_put_drop"] = int(st.dbg.get("jump_queue_put_drop", 0)) + 1
							except Exception:
								pass

					# History
					try:
						st.imu_history.append(
							{
								"t": t_i,
								"imu_timestamp": s.get("imu_timestamp"),
								"imu_sample_index": s.get("imu_sample_index"),
								"acc": s.get("acc", []),
								"gyro": s.get("gyro", []),
								"mag": s.get("mag", []),
							}
						)
						if len(st.imu_history) > int(CFG.runtime.imu_history_prune_check_min_len):
							cutoff = t_i - IMU_HISTORY_MAX_SECONDS
							while st.imu_history:
								first_t = float(st.imu_history[0].get("t", 0.0))
								if first_t >= cutoff:
									break
								st.imu_history.popleft()
					except Exception:
						pass

					# last_imu_t
					try:
						st.dbg["last_imu_t"] = float(t_i)
						st.dbg["imu_now_minus_t_s"] = float(time.time()) - float(t_i)
					except Exception:
						pass

					# Recording IMU CSV
					try:
						if st.imu_csv_fh is not None:
							acc = s.get("acc", []) or []
							gyro = s.get("gyro", []) or []
							mag = s.get("mag", []) or []
							ax, ay, az = (acc + [None, None, None])[:3]
							gx, gy, gz = (gyro + [None, None, None])[:3]
							mx, my, mz = (mag + [None, None, None])[:3]
							st.imu_csv_fh.write(
								f"{t_i},{s.get('imu_timestamp','')},{s.get('imu_sample_index','')},"
								f"{ax if ax is not None else ''},{ay if ay is not None else ''},{az if az is not None else ''},"
								f"{gx if gx is not None else ''},{gy if gy is not None else ''},{gz if gz is not None else ''},"
								f"{mx if mx is not None else ''},{my if my is not None else ''},{mz if mz is not None else ''}\n"
							)
					except Exception:
						pass

				# Broadcast to WS clients (preserve existing UI message shape).
				try:
					if st.manager:
						asyncio.create_task(
							st.manager.broadcast_json(
								{
									"t": time.time(),
									"mode": msg.get("mode"),
									"rate": msg.get("rate"),
									"seq": msg.get("seq"),
									"timestamp": msg.get("timestamp"),
									"samples_len": len(samples),
									"samples": [
										{"t": s.get("t"), "acc": s.get("acc", []), "gyro": s.get("gyro", []), "mag": s.get("mag", [])}
										for s in samples
									],
									"analysis": {},
									"first_sample": samples[0] if samples else None,
								}
							)
						)
				except Exception:
					pass

		transport, _ = await loop.create_datagram_endpoint(
			lambda: _ImuUdpProtocol(app_state),
			local_addr=(IMU_UDP_HOST, IMU_UDP_PORT),
		)
		app_state.imu_udp_transport = transport

		# Start jump clip worker process (file-queue based)
		try:
			jobs_dir = Path(JUMP_CLIP_JOBS_DIR)
			jobs_dir.mkdir(parents=True, exist_ok=True)
			cmd = [
				sys.executable,
				"-m",
				"modules.jump_clip_worker",
				"--jobs-dir",
				str(jobs_dir),
				"--poll-s",
				str(float(CFG.clip_worker.poll_seconds)),
			]
			log_path = jobs_dir / "worker.log"
			log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
			app_state.clip_worker_proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
			app_state.dbg["clip_worker_pid"] = int(app_state.clip_worker_proc.pid) if app_state.clip_worker_proc.pid else None
		except Exception:
			app_state.clip_worker_proc = None
			app_state.dbg["clip_worker_pid"] = None

		# Auto-connect: if default skater has IMU device, connect IMU and video; retry every 10s until IMU data arrives.
		app_state.auto_connect_task = asyncio.create_task(_auto_connect_loop(app_state))

		# Optionally start a dedicated video collector process (keeps camera/GPU code isolated).
		try:
			app_state.video_proc = start_video_collector_subprocess(CFG)
			app_state.dbg["video_proc_pid"] = int(app_state.video_proc.pid) if app_state.video_proc and app_state.video_proc.pid else None
		except Exception:
			app_state.video_proc = None
			app_state.dbg["video_proc_pid"] = None

		yield
	finally:
		# Use app_state for cleanup (same instance we created)
		st_clean = _app_state
		if st_clean is None:
			return
		# Cancel auto-connect task
		if st_clean.auto_connect_task:
			st_clean.auto_connect_task.cancel()
			try:
				await st_clean.auto_connect_task
			except asyncio.CancelledError:
				pass
			except Exception:
				pass
		st_clean.auto_connect_task = None

		# Stop video backend (release camera)
		try:
			if st_clean.video:
				st_clean.video.stop()
		except Exception:
			pass

		# Stop optional video collector process
		if st_clean.video_proc is not None:
			try:
				st_clean.video_proc.terminate()
			except Exception:
				pass
			try:
				st_clean.video_proc.wait(timeout=float(CFG.runtime.subprocess_wait_timeout_seconds))
			except Exception:
				pass
			st_clean.video_proc = None
			try:
				st_clean.dbg["video_proc_pid"] = None
			except Exception:
				pass

		# Stop jump clip worker process
		if st_clean.clip_worker_proc is not None:
			try:
				st_clean.clip_worker_proc.terminate()
			except Exception:
				pass
			try:
				st_clean.clip_worker_proc.wait(timeout=float(CFG.runtime.subprocess_wait_timeout_seconds))
			except Exception:
				pass
			st_clean.clip_worker_proc = None
			try:
				st_clean.dbg["clip_worker_pid"] = None
			except Exception:
				pass

		# Stop IMU collector process
		if st_clean.imu_proc is not None:
			try:
				st_clean.imu_proc.terminate()
			except Exception:
				pass
			try:
				st_clean.imu_proc.wait(timeout=float(CFG.runtime.subprocess_wait_timeout_seconds))
			except Exception:
				pass
			st_clean.imu_proc = None

		# Stop UDP receiver
		if st_clean.imu_udp_transport is not None:
			try:
				st_clean.imu_udp_transport.close()
			except Exception:
				pass
			st_clean.imu_udp_transport = None

		# Stop jump‑analysis worker
		if st_clean.jump_worker_task:
			st_clean.jump_worker_task.cancel()
			try:
				await st_clean.jump_worker_task
			except asyncio.CancelledError:
				pass
			except Exception:
				pass
		st_clean.jump_worker_task = None
		st_clean.jump_sample_queue = None

		# Stop frame sync task
		if st_clean.frame_sync_task:
			st_clean.frame_sync_task.cancel()
			try:
				await st_clean.frame_sync_task
			except asyncio.CancelledError:
				pass
			except Exception:
				pass
		st_clean.frame_sync_task = None


app = FastAPI(lifespan=lifespan)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# No HTML load at import (B.1): templates loaded lazily via get_page_html() when a page is requested.
# Mount routers before static so app routes take precedence; /static/* then served by StaticFiles (B.6).
from routers import pages, api_devices, api_skaters, api_coaches, api_jumps, video, sessions, ws
app.include_router(pages.router)

# Serve static assets (CSS, JS) from UI directory
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")
app.include_router(api_devices.router)
app.include_router(api_skaters.router)
app.include_router(api_coaches.router)
app.include_router(api_jumps.router)
app.include_router(video.router)
app.include_router(sessions.router)
app.include_router(ws.router)


def _log_to_clients(message: str) -> None:
	"""Send a log line to all connected WebSocket clients. Fire-and-forget; safe to call from non-async code."""
	try:
		mgr = _app_state.manager if _app_state else None
		if mgr:
			asyncio.create_task(mgr.broadcast_json({"type": "log", "msg": message}))
	except RuntimeError:
		pass


async def _frame_sync_loop(st: AppState) -> None:
	"""
	Periodic task that syncs frames from video backend to in-memory frame_history buffer.
	Runs continuously during server lifetime. Similar pattern to IMU buffering.
	
	The backend accumulates frames during recording in its own buffer; this loop
	periodically copies them to the server's frame_history for fast access by
	jump detection and clip generation code.
	"""
	_last_sync_frame_idx: int = -1  # Track by frame index to avoid timestamp drift issues
	
	while True:
		try:
			await asyncio.sleep(float(CFG.runtime.frame_sync_interval_seconds))
			
			# Only sync if recording is active
			if st.session_id is None:
				# No active session; clear history if it's stale
				_last_sync_frame_idx = -1
				# Don't clear history immediately; it might be needed for recent jumps
				continue
			
			# Get all frames from backend (since frame 0)
			# Backend's get_frames_since filters by t_host, but we track by index to avoid gaps
			all_backend_frames = []
			try:
				if st.video is not None:
					# Get frames from the start of current session (or all if not tracking session start)
					# We'll filter to only new ones using frame_idx
					all_backend_frames = st.video.get_frames_since(0.0, include_frame_idx=True, include_width_height=True)
			except Exception:
				# Backend may not implement get_frames_since (e.g., stub backends)
				pass
			
			# Add only new frames (those with frame_idx > _last_sync_frame_idx)
			new_count = 0
			for f in all_backend_frames:
				try:
					frame_idx = int(f.get("frame_idx", -1))
					if frame_idx <= _last_sync_frame_idx:
						continue
					st.frame_history.append(f)
					_last_sync_frame_idx = max(_last_sync_frame_idx, frame_idx)
					new_count += 1
				except Exception:
					continue
			
			# Prune old frames (keep last FRAME_HISTORY_MAX_SECONDS)
			try:
				if st.frame_history:
					cutoff = time.time() - FRAME_HISTORY_MAX_SECONDS
					# Since frames are added in order, we can prune from the front
					while st.frame_history:
						first_t = float(st.frame_history[0].get("t_host", 0.0))
						if first_t >= cutoff:
							break
						# If we prune, reset sync index to avoid gaps
						pruned_idx = int(st.frame_history[0].get("frame_idx", -1))
						if pruned_idx >= 0:
							_last_sync_frame_idx = max(-1, pruned_idx - 1)
						st.frame_history.popleft()
			except Exception:
				pass
				
		except asyncio.CancelledError:
			break
		except Exception:
			# Keep loop running even on errors
			await asyncio.sleep(float(CFG.runtime.frame_sync_error_backoff_seconds))


def _ensure_unique_jump_window(
	st: AppState,
	session_id: Optional[str],
	event_id: int,
	base_window_start: float,
	base_window_end: float,
	t_peak: float,
) -> Tuple[float, float]:
	"""
	Ensure jump window is unique by checking for overlaps with existing jumps
	in the same session and adjusting boundaries if needed.
	
	Returns: (adjusted_window_start, adjusted_window_end)
	"""
	if not session_id:
		# No session tracking, return base window
		return (base_window_start, base_window_end)
	
	# Get existing windows for this session
	existing_windows = st.jump_windows_by_session.get(session_id, [])
	
	window_start = base_window_start
	window_end = base_window_end
	
	# Check for overlaps and adjust
	overlap_tolerance = float(CFG.runtime.overlap_tolerance_seconds)
	for existing_event_id, existing_start, existing_end in existing_windows:
		if existing_event_id == event_id:
			continue  # Skip self
		
		# Check if windows overlap
		if not (window_end < existing_start - overlap_tolerance or window_start > existing_end + overlap_tolerance):
			# Overlap detected - adjust window to be unique
			# Strategy: split the difference between the two jumps
			overlap_start = max(window_start, existing_start)
			overlap_end = min(window_end, existing_end)
			overlap_center = (overlap_start + overlap_end) / 2.0
			
			# If this jump's peak is before the overlap center, adjust end boundary
			# Otherwise, adjust start boundary
			if t_peak < overlap_center:
				# This jump is earlier - adjust end to be before existing start
				window_end = min(window_end, existing_start - overlap_tolerance)
			else:
				# This jump is later - adjust start to be after existing end
				window_start = max(window_start, existing_end + overlap_tolerance)
			
			# Ensure window still includes the jump (minimum window size)
			min_window_size = float(CFG.runtime.min_window_size_seconds)
			if window_end - window_start < min_window_size:
				# If adjustment made window too small, center it on t_peak
				window_center = t_peak
				window_start = window_center - min_window_size / 2.0
				window_end = window_center + min_window_size / 2.0
			
			# Ensure window_start < window_end
			if window_start >= window_end:
				# Fallback: use a tight window around t_peak
				half = float(CFG.runtime.fallback_window_half_seconds)
				window_start = t_peak - half
				window_end = t_peak + half
	
	# Store this window for future overlap checks
	if session_id not in st.jump_windows_by_session:
		st.jump_windows_by_session[session_id] = []
	st.jump_windows_by_session[session_id].append((event_id, window_start, window_end))
	
	# Clean up old windows (keep last 100 per session to avoid unbounded growth)
	if len(st.jump_windows_by_session[session_id]) > int(CFG.runtime.jump_windows_history_per_session):
		st.jump_windows_by_session[session_id] = st.jump_windows_by_session[session_id][-int(CFG.runtime.jump_windows_history_per_session):]
	
	return (window_start, window_end)


async def _jump_worker_loop(st: AppState) -> None:
	"""
	Background task that consumes IMU samples from the queue and runs
	JumpDetectorRealtime to emit structured jump events, decoupled from BLE.
	"""
	# One detector instance per worker, recreated when stream rate changes.
	jump_detector: Optional[JumpDetectorRealtime] = None
	last_rate: Optional[int] = None

	while True:
		if st.jump_sample_queue is None:
			# Should not happen often; be defensive.
			await asyncio.sleep(float(CFG.runtime.jump_worker_idle_sleep_seconds))
			continue

		sample = await st.jump_sample_queue.get()
		try:
			st.dbg["jump_worker_samples"] = int(st.dbg.get("jump_worker_samples", 0)) + 1
		except Exception:
			pass
		# Lazily create (or recreate) detector with the active stream rate.
		active_rate = int(st.active_rate) if int(st.active_rate) > 0 else RATE
		if jump_detector is None or last_rate != active_rate:
			jump_detector = JumpDetectorRealtime(
				sample_rate_hz=active_rate,
				window_seconds=float(st.jump_config.get("window_seconds", CFG.jump_detection.window_seconds)),
				logger=_jump_log_filter,
				config=st.jump_config,
			)
			last_rate = active_rate
		try:
			events = (jump_detector.update(sample) if jump_detector else []) or []
		except Exception:
			events = []

		if not events:
			continue

		# Only emit jump messages once detection has been explicitly enabled.
		if not st.jump_detection_enabled:
			continue

		for ev in events:
			try:
				event_id = st.next_event_id
				st.next_event_id += 1
				try:
					st.dbg["jump_events_emitted"] = int(st.dbg.get("jump_events_emitted", 0)) + 1
					st.dbg["last_jump_t"] = float(ev.get("t_peak") or time.time())
					st.dbg["last_jump_event_id"] = int(event_id)
				except Exception:
					pass

				# Ensure t_peak is a valid epoch-seconds float (DB requires NOT NULL).
				t_peak_val = ev.get("t_peak")
				if t_peak_val is None:
					# Fallback: use current time to avoid None propagating downstream.
					t_peak_val = time.time()
				try:
					t_peak_val = float(t_peak_val)
				except (TypeError, ValueError):
					t_peak_val = time.time()

				jump_msg = {
					"type": "jump",
					"event_id": event_id,
					"t_peak": t_peak_val,
					# Also expose refined takeoff/landing times so the UI can align markers to
					# "feet leave ground" and "feet touch ground" rather than the rotation peak.
					"t_takeoff": ev.get("t_takeoff"),
					"t_landing": ev.get("t_landing"),
					# Server-side emission time (helps quantify perceived delay).
					"t_emitted": float(time.time()),
					"flight_time": ev.get("flight_time"),
					"height": ev.get("height"),
					"acc_peak": ev.get("peak_az_no_g"),
					"gyro_peak": ev.get("peak_gz"),
					"rotation_phase": ev.get("rotation_phase"),
					"confidence": ev.get("confidence"),
					# Step 2.2/2.3: revolutions and under-rotation
					"revolutions_est": ev.get("revolutions_est"),
					"revolutions_class": ev.get("revolutions_class"),
					"underrotation": ev.get("underrotation"),
					"underrot_flag": ev.get("underrot_flag"),
				}
				# Diagnostic: emission latency relative to the detected peak timestamp.
				try:
					st.dbg["jump_emit_delay_s"] = float(time.time()) - float(t_peak_val)
					if ev.get("t_takeoff") is not None:
						st.dbg["jump_emit_delay_from_takeoff_s"] = float(time.time()) - float(ev.get("t_takeoff"))  # type: ignore[arg-type]
				except Exception:
					pass

				# Keep a compact record for export / labelling.
				record = dict(jump_msg)
				record["t_takeoff"] = ev.get("t_takeoff")
				record["t_landing"] = ev.get("t_landing")
				st.jump_events.append(record)

				# Ensure there is at least a stub annotation (name can be overridden later).
				if event_id not in st.jump_annotations:
					st.jump_annotations[event_id] = {"name": f"Jump {event_id}", "note": None}
				asyncio.create_task(st.manager.broadcast_json(jump_msg))

				# Fire‑and‑forget persistence of this jump and its IMU window to DB.
				try:
					# Use configurable window around t_peak for DB storage.
					t_peak_val = float(ev.get("t_peak", 0.0))
					t_takeoff_val = float(ev.get("t_takeoff", t_peak_val))
					t_landing_val = float(ev.get("t_landing", t_peak_val))
					pre_jump_s = float(st.cfg.jump_recording.pre_jump_seconds)
					post_jump_s = float(st.cfg.jump_recording.post_jump_seconds)
					# Window should be centered on t_peak, but ensure it includes takeoff/landing
					edge_guard = float(CFG.runtime.takeoff_landing_edge_guard_seconds)
					base_window_start = min(t_takeoff_val - edge_guard, t_peak_val - pre_jump_s)
					base_window_end = max(t_landing_val + edge_guard, t_peak_val + post_jump_s)
					
					active_session_id = st.session_id or st.detection_session_id

					# Ensure window uniqueness: check for overlaps with existing jumps in same session
					# and adjust boundaries to avoid overlap
					window_start, window_end = _ensure_unique_jump_window(
						st,
						active_session_id,
						event_id,
						base_window_start,
						base_window_end,
						t_peak_val,
					)
					ann = st.jump_annotations.get(event_id) or {}
					jump_for_db = dict(record)
					# Link to current recording session (if any) so jumps can be played back.
					# Capture at scheduling time to avoid race with later state mutations.
					jump_for_db["session_id"] = active_session_id
					jump_for_db["t_start"] = window_start
					jump_for_db["t_end"] = window_end
					# Persist immediately on detection. Any heavy post-processing (ffmpeg clip cutting,
					# jump_frames generation) is offloaded to a separate worker process via job file.
					#
					# IMPORTANT: bind values as explicit function arguments.
					# This avoids a classic late-binding closure bug where multiple create_task(...)
					# calls inside the loop would all see the "last" event_id/jump_for_db/window values.
					async def _persist_and_enqueue(
						st_arg: AppState,
						jump_for_db_arg: Dict[str, Any],
						ann_arg: Dict[str, Any],
						window_start_arg: float,
						window_end_arg: float,
						event_id_arg: int,
						t_peak_val_arg: float,
						session_id_arg: Optional[str],
					) -> None:
						# Critical: at t_peak time we usually DO NOT yet have the future IMU samples.
						# Wait briefly for the stream to advance so we can persist the full requested window.
						# Optimized: reduce wait time since we have in-memory buffer, and use more aggressive polling.
						try:
							post_jump_s = float(st_arg.cfg.jump_recording.post_jump_seconds)
							deadline = time.time() + post_jump_s + float(st_arg.cfg.runtime.imu_wait_slack_seconds)
							wait_start = time.time()
							while time.time() < deadline:
								last_t = st_arg.dbg.get("last_imu_t")
								try:
									last_t_f = float(last_t) if last_t is not None else None
								except Exception:
									last_t_f = None
								if last_t_f is not None and last_t_f >= float(window_end_arg):
									break
								await asyncio.sleep(float(st_arg.cfg.runtime.imu_wait_poll_seconds))
							try:
								st_arg.dbg["db_window_waited_s"] = float(max(0.0, time.time() - wait_start))
							except Exception:
								pass
						except Exception:
							pass

						# Slice history AFTER waiting, so we include samples up through window_end.
						window_samples = []
						try:
							history_list = list(st_arg.imu_history) if st_arg.imu_history else []
							if history_list:
								times = [float(row.get("t", 0.0)) for row in history_list]
								left_idx = bisect.bisect_left(times, window_start_arg)
								right_idx = bisect.bisect_right(times, window_end_arg)
								window_samples = history_list[left_idx:right_idx]
						except Exception:
							window_samples = []

						try:
							jump_id = await db.insert_jump_with_imu(jump_for_db_arg, ann_arg, window_samples)
							if not jump_id:
								st_arg.dbg["db_last_insert_error"] = "insert_jump_with_imu returned no jump_id (DB disabled or insert skipped)"
								return
							st_arg.dbg["db_last_insert_ok_t"] = float(time.time())
							st_arg.dbg["db_last_insert_error"] = None
							st_arg.log_to_clients(
								f"[DB] Inserted jump: event_id={event_id_arg}, jump_id={jump_id}, samples={len(window_samples)} "
								f"(window={window_start_arg:.3f}->{window_end_arg:.3f})"
							)
							# Notify websocket clients only after jump row is persisted.
							if st_arg.manager is not None:
								asyncio.create_task(
									st_arg.manager.broadcast_json(
										{
											"type": "jump_saved",
											"event_id": int(event_id_arg),
											"jump_id": int(jump_id),
											"session_id": str(session_id_arg or ""),
											"t_peak": float(t_peak_val_arg),
										}
									)
								)
						except Exception as e:
							st_arg.dbg["db_last_insert_error"] = repr(e)
							st_arg.log_to_clients(f"[DB] insert_jump_with_imu failed for event_id={event_id_arg}: {e!r}")
							logger.error("[DB] insert_jump_with_imu failed for event_id=%s: %s", event_id_arg, e)
							return

						# Enqueue clip generation job. This runs out-of-process and may complete later
						# (e.g., after MP4 mux is available).
						try:
							clip_buffer_s = float(st_arg.cfg.jump_recording.clip_buffer_seconds)
							clip_start_host = float(window_start_arg) - clip_buffer_s
							clip_end_host = float(window_end_arg) + clip_buffer_s
							clip_duration = clip_end_host - clip_start_host
							base_dir = Path(st_arg.cfg.sessions.base_dir) / str(session_id_arg or "")
							mp4_path = base_dir / "video.mp4"
							h264_path = base_dir / "video.h264"
							# Debug logging for clip length issues
							st_arg.log_to_clients(
								f"[Clip] event_id={event_id_arg}: window=[{window_start_arg:.3f}, {window_end_arg:.3f}], "
								f"clip=[{clip_start_host:.3f}, {clip_end_host:.3f}], duration={clip_duration:.2f}s"
							)
						except Exception:
							pre_jump_s = float(st_arg.cfg.jump_recording.pre_jump_seconds)
							post_jump_s = float(st_arg.cfg.jump_recording.post_jump_seconds)
							clip_extra = float(CFG.runtime.clip_fallback_extra_seconds)
							clip_start_host = float(t_peak_val_arg) - pre_jump_s - clip_extra
							clip_end_host = float(t_peak_val_arg) + post_jump_s + clip_extra

						st_arg.enqueue_jump_clip_job(
							{
								"jump_id": int(jump_id),
								"event_id": int(event_id_arg),
								"session_id": str(session_id_arg or ""),
								"clip_start_host": float(clip_start_host),
								"clip_end_host": float(clip_end_host),
								"video_fps": int(st_arg.cfg.video.recording_fps),
								"wait_mp4_timeout_s": float(CFG.runtime.clip_wait_mp4_timeout_seconds),
							}
						)

					try:
						ws_ = float(window_start)
						we_ = float(window_end)
					except Exception:
						pre_jump_s = float(st.cfg.jump_recording.pre_jump_seconds)
						post_jump_s = float(st.cfg.jump_recording.post_jump_seconds)
						ws_ = float(t_peak_val) - pre_jump_s
						we_ = float(t_peak_val) + post_jump_s
					try:
						tp_ = float(t_peak_val)
					except Exception:
						tp_ = time.time()
					asyncio.create_task(
						_persist_and_enqueue(
							st,
							dict(jump_for_db),
							dict(ann),
							ws_,
							we_,
							int(event_id),
							tp_,
							active_session_id,
						)
					)
				except Exception as e:
					# DB persistence is best‑effort; ignore errors here.
					logger.error("[DB] Error scheduling insert_jump_with_imu: %s", e)
			except Exception:
				# Jump notifications are best‑effort; ignore errors.
				pass
def _session_base_dir(session_id: str) -> Path:
	# Session directories are stored under <sessions.base_dir>/<session_id>/
	# NOTE: Do not call get_config() here because server.py defines an async route handler
	# named get_config(), which can shadow the imported modules.config.get_config symbol.
	# Use the already-loaded global CFG instead.
	base = Path(CFG.sessions.base_dir)
	return base / session_id


def _enqueue_jump_clip_job(job: Dict[str, Any]) -> None:
	"""
	Enqueue a jump clip job by writing a small JSON file into JUMP_CLIP_JOBS_DIR.
	This is intentionally file-based so it survives restarts and doesn't block the realtime loop.
	"""
	try:
		jobs_dir = Path(JUMP_CLIP_JOBS_DIR)
		jobs_dir.mkdir(parents=True, exist_ok=True)
		jid = int(job.get("jump_id") or 0)
		ts = int(time.time() * 1000)
		p = jobs_dir / f"jump_{jid}_{ts}.json"
		p.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
	except Exception:
		pass


def _count_clip_jobs_pending() -> int:
	try:
		jobs_dir = Path(JUMP_CLIP_JOBS_DIR)
		return len(list(jobs_dir.glob("*.json")))
	except Exception:
		return 0


def _count_clip_jobs_done() -> int:
	try:
		jobs_dir = Path(JUMP_CLIP_JOBS_DIR) / "done"
		return len(list(jobs_dir.glob("*.done.json")))
	except Exception:
		return 0


def _count_clip_jobs_failed() -> int:
	try:
		jobs_dir = Path(JUMP_CLIP_JOBS_DIR) / "failed"
		return len(list(jobs_dir.glob("*.failed.json")))
	except Exception:
		return 0


def _read_last_clip_job_error() -> Optional[str]:
	"""
	Best-effort: read the newest *.error.txt under jobs/failed and return its first line.
	"""
	try:
		fail_dir = Path(JUMP_CLIP_JOBS_DIR) / "failed"
		if not fail_dir.exists():
			return None
		errs = sorted(fail_dir.glob("*.error.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
		if not errs:
			return None
		txt = errs[0].read_text(encoding="utf-8", errors="ignore").strip().splitlines()
		return txt[0] if txt else None
	except Exception:
		return None


def _load_frames_start_host(session_dir: Path) -> Optional[float]:
	"""
	Read frames.csv and return the first frame's t_host (epoch seconds) if available.
	"""
	p = session_dir / "frames.csv"
	if not p.exists():
		return None
	try:
		with open(p, "r", encoding="utf-8") as fh:
			_ = fh.readline()  # header
			for line in fh:
				parts = line.strip().split(",")
				if len(parts) >= 2 and parts[1]:
					return float(parts[1])
	except Exception:
		return None
	return None


async def _generate_jump_clips_for_session(session_id: str) -> None:
	"""
	Best-effort: for all jumps in this session, cut a per-jump mp4 clip and
	store its relative path to DB (jumps.video_path).
	"""
	try:
		from modules.video_backend import get_video_backend_for_tools

		base = _session_base_dir(session_id)
		mp4 = base / "video.mp4"
		if not mp4.exists():
			return
		start_host = _load_frames_start_host(base)
		if start_host is None:
			return

		# Pull all jumps and filter locally by session_id (keeps DB changes minimal).
		rows = await db.list_jumps(limit=int(CFG.runtime.backfill_jumps_limit))
		jumps_in_session = [r for r in rows if r.get("session_id") == session_id]
		if not jumps_in_session:
			return

		clip_backend = get_video_backend_for_tools()

		clips_dir = base / CFG.sessions.jump_clips_subdir
		clips_dir.mkdir(parents=True, exist_ok=True)

		# Load session frames (from DB if present, otherwise from file endpoint fallback).
		# We'll use these to store per-jump clip-relative frame timing linked to each jump.
		session_frames: list[dict] = []
		try:
			session_frames = await db.get_frames(session_id=session_id, limit=int(CFG.runtime.backfill_frames_limit))
		except Exception:
			session_frames = []
		if not session_frames:
			# Fallback to parsing frames.csv directly
			try:
				p = base / "frames.csv"
				if p.exists():
					with open(p, "r", encoding="utf-8") as fh:
						_ = fh.readline()
						for line in fh:
							parts = line.strip().split(",")
							if len(parts) < 5:
								continue
							try:
								session_frames.append(
									{
										"frame_idx": int(parts[0]),
										"t_host": float(parts[1]),
										"device_ts": float(parts[2]) if parts[2] else None,
										"width": int(parts[3]) if parts[3] else None,
										"height": int(parts[4]) if parts[4] else None,
									}
								)
							except Exception:
								continue
			except Exception:
				session_frames = []

		for j in jumps_in_session:
			try:
				event_id = int(j.get("event_id") or 0)
				if event_id <= 0:
					continue
				if j.get("video_path"):
					continue

				t0 = j.get("t_start")
				t1 = j.get("t_end")
				tp = j.get("t_peak")
				pre = float(CFG.jump_recording.clip_buffer_seconds)
				post = float(CFG.jump_recording.clip_buffer_seconds)
				if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and float(t1) > float(t0):
					clip_start_host = float(t0) - pre
					clip_end_host = float(t1) + post
				elif isinstance(tp, (int, float)):
					half = float(CFG.export.jump_fallback_half_window_seconds)
					clip_start_host = float(tp) - half
					clip_end_host = float(tp) + half
				else:
					continue

				start_sec = max(0.0, clip_start_host - start_host)
				duration = max(0.2, clip_end_host - clip_start_host)

				out_name = f"jump_{event_id}.mp4"
				out_path = clips_dir / out_name

				ok_cut = False
				try:
					ok_cut = bool(clip_backend.cut_clip_best_effort(mp4, out_path, start_sec=start_sec, duration_sec=duration))
				except Exception:
					ok_cut = False

				if ok_cut and out_path.exists() and out_path.stat().st_size > 0:
					# Store a relative path usable by GET /files (restricted to data/ subtree).
					rel = str(Path(CFG.sessions.base_dir) / session_id / CFG.sessions.jump_clips_subdir / out_name)
					await db.set_jump_video_path(event_id, rel)

				# Persist per-jump frames (clip-relative) linked to jump_id.
				try:
					jump_id = int(j.get("jump_id") or 0)
				except Exception:
					jump_id = 0
				if jump_id > 0 and session_frames:
					# Select frames within the clip host window and map to clip time (t_video)
					cut0 = clip_start_host
					cut1 = clip_end_host
					selected = [f for f in session_frames if isinstance(f.get("t_host"), (int, float)) and float(f["t_host"]) >= cut0 and float(f["t_host"]) <= cut1]
					# Re-index frames from 0..N-1 and compute t_video from host time
					jf = []
					for i, f in enumerate(selected):
						th = float(f["t_host"])
						jf.append(
							{
								"frame_idx": i,
								"t_video": max(0.0, th - cut0),
								"t_host": th,
								"device_ts": f.get("device_ts"),
								"width": f.get("width"),
								"height": f.get("height"),
							}
						)
					await db.replace_jump_frames(jump_id, jf)
			except Exception:
				continue
	except Exception:
		return


@app.get("/files")
async def get_file(path: str):
	"""
	Serve a file under the workspace (restricted to data/ only).
	Used for per-jump clip playback where DB stores a relative path like:
	  data/sessions/<sid>/jump_clips/jump_<event_id>.mp4
	"""
	if not path:
		raise HTTPException(status_code=400, detail="Missing path")
	p = Path(path)
	# Restrict to data/ subtree
	try:
		p_resolved = p.resolve()
		data_resolved = Path("data").resolve()
		if data_resolved not in p_resolved.parents and p_resolved != data_resolved:
			raise HTTPException(status_code=403, detail="Forbidden path")
	except HTTPException:
		raise
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid path")
	if not p.exists():
		raise HTTPException(status_code=404, detail="File not found")
	# Infer by extension
	media_type = "application/octet-stream"
	if p.suffix.lower() == ".mp4":
		media_type = "video/mp4"
	return FileResponse(str(p), media_type=media_type, filename=p.name)


@app.post("/connect")
async def connect_device(payload: ConnectPayload, state: AppState = Depends(get_state)):
	"""
	Start the IMU collector process connected to the specified device (MAC or name).
	The collector streams IMU packets to this server over localhost UDP.
	If a collector is already running, it will be stopped and replaced.

	Can accept either:
	- "device": MAC address or device name
	- "skater_id": skater ID (will use first registered device for that skater)
	"""
	try:
		device = (payload.device or "").strip()
		skater_id = payload.skater_id
		mode = (payload.mode or MODE).strip().upper()
		rate = int(payload.rate or RATE)
		state.active_rate = rate
		state.active_mode = mode

		# If skater_id is provided, get the first device for that skater
		if skater_id:
			skater_devices = await db.get_skater_devices(int(skater_id))
			if not skater_devices:
				raise HTTPException(status_code=400, detail="No sensor registered for this skater")
			waist_device = next((d for d in skater_devices if d.get("placement") == "waist"), None)
			selected_device = waist_device or skater_devices[0]
			device = selected_device.get("mac_address")
			if not device:
				raise HTTPException(status_code=400, detail="Device MAC address not found")
		elif not device:
			raise HTTPException(status_code=400, detail="Missing 'device' or 'skater_id' in request body.")

		resolved_mac = await db.resolve_device_identifier(device)
		if resolved_mac:
			device = resolved_mac

		if state.imu_proc is not None:
			try:
				state.imu_proc.terminate()
			except Exception:
				pass
			try:
				state.imu_proc.wait(timeout=float(CFG.runtime.subprocess_wait_timeout_seconds))
			except Exception:
				pass
			state.imu_proc = None

		cmd = [
			sys.executable,
			"-m",
			"modules.imu_collector",
			"--device",
			device,
			"--mode",
			mode,
			"--rate",
			str(rate),
			"--udp-host",
			IMU_UDP_HOST,
			"--udp-port",
			str(IMU_UDP_PORT),
		]
		state.imu_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		try:
			state.dbg["collector_running"] = True
			state.dbg["collector_pid"] = int(state.imu_proc.pid) if state.imu_proc.pid else None
		except Exception:
			pass

		state.log_to_clients(f"[Collector] Started PID={state.imu_proc.pid} (mode={mode}, rate={rate}, udp={IMU_UDP_HOST}:{IMU_UDP_PORT})")
		return {"detail": f"Starting IMU collector for {device} (mode={mode}, rate={rate})", "collector_pid": state.imu_proc.pid}
	except HTTPException:
		raise
	except Exception as e:
		state.log_to_clients(f"/connect error: {e!r}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/disconnect")
async def disconnect_device(state: AppState = Depends(get_state)):
	"""
	Stop the current IMU collector process (if any).
	"""
	if state.imu_proc is not None:
		try:
			state.imu_proc.terminate()
		except Exception:
			pass
		try:
			state.imu_proc.wait(timeout=float(CFG.runtime.subprocess_wait_timeout_seconds))
		except Exception:
			pass
		state.imu_proc = None
		try:
			state.dbg["collector_running"] = False
			state.dbg["collector_pid"] = None
		except Exception:
			pass
	state.log_to_clients("IMU collector stopped via /disconnect")
	return {"detail": "Stopped IMU collector"}


@app.post("/detection/start")
async def start_jump_detection(state: AppState = Depends(get_state)):
	"""
	Manually enable jump detection, to avoid false positives during sensor
	setup / strapping on.
	"""
	state.jump_detection_enabled = True
	state.log_to_clients("[JumpDetector] Detection ENABLED via /detection/start")

	if state.detection_session_id is None:
		sid = time.strftime("%Y%m%d_%H%M%S") + "_detect"
		try:
			resp = await session_start(SessionStartPayload(session_id=sid), state=state)
			resolved_sid = (
				(resp.get("session_id") if isinstance(resp, dict) else None)
				or state.session_id
				or sid
			)
			state.detection_session_id = str(resolved_sid)
			state.log_to_clients(f"[Session] Auto-started recording session for detection: {state.detection_session_id}")
		except HTTPException as e:
			state.detection_session_id = None
			state.log_to_clients(f"[Session] Auto-start for detection failed: {e.detail}")
			logger.error("[Session] Auto-start for detection failed: %s", e.detail)
			raise
		except Exception as e:
			state.detection_session_id = None
			state.log_to_clients(f"[Session] Auto-start for detection failed: {e!r}")
			logger.error("[Session] Auto-start for detection failed: %s", e)
			raise HTTPException(status_code=500, detail=f"Auto-start recording failed: {e!r}")

	return {"detail": "Jump detection enabled (recording auto-started).", "session_id": state.detection_session_id}


@app.post("/detection/stop")
async def stop_jump_detection(state: AppState = Depends(get_state)):
	"""
	Manually disable jump detection while keeping BLE streaming active.
	"""
	state.jump_detection_enabled = False
	state.log_to_clients("[JumpDetector] Detection DISABLED via /detection/stop")
	try:
		if state.detection_session_id is not None:
			await session_stop(state=state)
			state.log_to_clients(f"[Session] Auto-stopped recording session for detection: {state.detection_session_id}")
	finally:
		state.detection_session_id = None
	return {"detail": "Jump detection disabled (recording auto-stopped)."}


@app.get("/annotations/{event_id}")
async def get_annotation(event_id: int, state: AppState = Depends(get_state)):
	"""
	Fetch stored annotation (name/note, later labels) for a jump event.
	Returns {} if none exists.
	"""
	return state.jump_annotations.get(event_id, {})


@app.post("/annotations/{event_id}")
async def set_annotation(event_id: int, payload: Dict[str, Any], state: AppState = Depends(get_state)):
	"""
	Store or update annotation for a jump event (name, note, future label fields).
	"""
	data = payload or {}
	current = state.jump_annotations.get(event_id, {})
	current.update(
		{
			"name": data.get("name", current.get("name")),
			"note": data.get("note", current.get("note")),
		}
	)
	state.jump_annotations[event_id] = current
	# Persist to DB as best‑effort (if configured).
	try:
		asyncio.create_task(
			db.update_annotation(
				event_id=event_id,
				name=current.get("name"),
				note=current.get("note"),
			)
		)
	except Exception:
		# DB persistence of annotations is best‑effort.
		pass
	return {"event_id": event_id, "annotation": current}


@app.post("/annotations/by_jump_id/{jump_id}")
async def set_annotation_by_jump_id(jump_id: int, payload: Dict[str, Any]):
	"""
	Store or update annotation for a specific DB jump row (jump_id).

	This avoids the ambiguity of event_id which is not guaranteed unique across sessions.
	"""
	data = payload or {}
	name = data.get("name")
	note = data.get("note")
	try:
		await db.update_annotation_by_jump_id(
			jump_id=int(jump_id),
			name=(str(name) if name is not None else None),
			note=(str(note) if note is not None else None),
		)
	except Exception:
		pass
	return {"jump_id": int(jump_id), "annotation": {"name": name, "note": note}}


@app.get("/config")
async def get_config_route(state: AppState = Depends(get_state)):
	"""
	Return current jump detection configuration.
	"""
	return {"jump": state.jump_config}


@app.get("/api/version")
async def get_app_version():
	"""Return product name + software version for UI and tooling."""
	return {"name": APP_NAME, "version": _read_app_version()}


@app.post("/config")
async def set_config(payload: Dict[str, Any], state: AppState = Depends(get_state)):
	"""
	Update jump detection configuration. Changes take effect on the next
	connection (when a new JumpDetectorRealtime is created).
	DEPRECATED: Use /api/skaters/{skater_id}/detection-settings instead.
	"""
	body = payload or {}
	jump_cfg = body.get("jump") or {}
	if not isinstance(jump_cfg, dict):
		raise HTTPException(status_code=400, detail="Field 'jump' must be an object")

	for key, value in jump_cfg.items():
		if key in JUMP_CONFIG_DEFAULTS:
			try:
				state.jump_config[key] = float(value)
			except (TypeError, ValueError):
				continue

	return {"detail": "Jump config updated. Reconnect sensor to apply."}


def _compute_analysis(sample: Dict[str, Any]) -> Dict[str, float]:
	# Simple derived metrics for demo
	ax, ay, az = sample["acc"]
	acc_mag = (ax*ax + ay*ay + az*az) ** 0.5
	gx, gy, gz = sample["gyro"]
	gyro_rms = ((gx*gx + gy*gy + gz*gz) / 3.0) ** 0.5
	out = {"acc_mag_first": float(acc_mag), "gyro_rms_first": float(gyro_rms)}
	if "mag" in sample and sample["mag"]:
		mx, my, mz = sample["mag"]
		mag_mag = (mx*mx + my*my + mz*mz) ** 0.5
		out["mag_mag_first"] = float(mag_mag)
	return out

@app.get("/export")
async def export_imu(seconds: Optional[float] = None, mode: str = "raw", state: AppState = Depends(get_state)):
	"""
	Export recent IMU data for offline analysis / labelling.

	Modes:
	  - mode="raw" (default): return IMU samples from the last `seconds`.
	  - mode="jumps": return per‑jump windows keyed by event_id, using
	    detected jump events within the last `seconds`.
	"""
	now = time.time()
	if seconds is None:
		seconds = float(CFG.export.default_seconds)
	horizon = max(0.0, float(seconds))
	cutoff = now - horizon

	history_snapshot = list(state.imu_history) if state.imu_history else []

	if mode == "jumps":
		events_snapshot = list(state.jump_events) if state.jump_events else []
		windows = []
		for ev in events_snapshot:
			t_peak = float(ev.get("t_peak", 0.0))
			if t_peak < cutoff:
				continue
			t0 = ev.get("t_takeoff")
			t1 = ev.get("t_landing")
			if t0 is None or t1 is None:
				half = float(CFG.export.jump_fallback_half_window_seconds)
				t0 = t_peak - half
				t1 = t_peak + half
			t0 = float(t0)
			t1 = float(t1)
			if t1 <= t0:
				continue

			samples = [
				row
				for row in history_snapshot
				if t0 <= float(row.get("t", 0.0)) <= t1
			]

			ann = state.jump_annotations.get(int(ev.get("event_id", 0))) or {}

			windows.append(
				{
					"event_id": ev.get("event_id"),
					"t_peak": t_peak,
					"t_start": t0,
					"t_end": t1,
					"meta": {
						"flight_time": ev.get("flight_time"),
						"height": ev.get("height"),
						"acc_peak": ev.get("acc_peak"),
						"gyro_peak": ev.get("gyro_peak"),
						"rotation_phase": ev.get("rotation_phase"),
						"confidence": ev.get("confidence"),
						"revolutions_est": ev.get("revolutions_est"),
						"revolutions_class": ev.get("revolutions_class"),
						"underrotation": ev.get("underrotation"),
						"underrot_flag": ev.get("underrot_flag"),
						"name": ann.get("name"),
						"note": ann.get("note"),
					},
					"samples": samples,
				}
			)

		return {"mode": "jumps", "seconds": seconds, "count": len(windows), "windows": windows}

	# Default: raw export of recent IMU samples.
	samples = [row for row in history_snapshot if float(row.get("t", 0.0)) >= cutoff]
	return {"mode": "raw", "seconds": seconds, "count": len(samples), "samples": samples}