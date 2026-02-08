import asyncio
import json
import os
import sys
import time
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
from schemas.requests import ConnectPayload
import state

# Reuse your BLE client
from modules.movesense_gatt import MovesenseGATTClient
from modules import db
from modules.jump_detector import JumpDetectorRealtime
from modules.video_backend import get_video_backend, start_video_collector_subprocess
from modules.config import get_config

# ----------------------------
# Pose auto-run (best-effort)
# ----------------------------
_pose_jobs_inflight: Set[int] = set()


async def _run_pose_for_jump_best_effort(event_id: int, max_fps: float = 10.0) -> Dict[str, Any]:
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
			max_fps = float(max_fps)
		except Exception:
			max_fps = 10.0
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

		provider = MediaPipePoseProvider()
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
	if ev in _pose_jobs_inflight:
		return
	_pose_jobs_inflight.add(ev)

	async def _runner():
		try:
			await _run_pose_for_jump_best_effort(ev, max_fps=10.0)
		finally:
			try:
				_pose_jobs_inflight.discard(ev)
			except Exception:
				pass

	try:
		asyncio.create_task(_runner())
	except Exception:
		try:
			_pose_jobs_inflight.discard(ev)
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

# IMU collector process globals (separate process architecture)
IMU_UDP_HOST = (CFG.imu_udp.host or "127.0.0.1").strip()
IMU_UDP_PORT = int(CFG.imu_udp.port or 9999)
_imu_proc: Optional[subprocess.Popen] = None
_imu_udp_transport: Optional[asyncio.DatagramTransport] = None

# Jump clip worker process (offloads ffmpeg + heavy DB work from realtime detection)
JUMP_CLIP_JOBS_DIR = (CFG.jobs.jump_clip_jobs_dir or str(Path("data") / "jobs" / "jump_clips"))
_clip_worker_proc: Optional[subprocess.Popen] = None

# Active stream settings (set on /connect). Used by analysis worker.
_active_mode: str = MODE
_active_rate: int = RATE

# Video backend (Raspberry Pi Camera Module 3 / CSI today; later Jetson/GStreamer).
_video = get_video_backend(CFG)  # primary (Module 3)
_video_proc: Optional[subprocess.Popen] = None

# Session recording (video + IMU) for Phase A3
_session_lock = asyncio.Lock()
_session_id: Optional[str] = None
_session_dir: Optional[Path] = None
_imu_csv_fh: Optional[Any] = None
_detection_session_id: Optional[str] = None

# Jump detection config (Phase 2.5/2.6): tweakable via /config endpoint.
JUMP_CONFIG_DEFAULTS: Dict[str, float] = {
	"min_jump_height_m": 0.15,
	"min_jump_peak_az_no_g": 3.5,
	"min_jump_peak_gz_deg_s": 180.0,
	"min_new_event_separation_s": 0.5,
	# How often JumpDetectorRealtime runs its heavier window/metrics pass (seconds).
	# Lower -> more responsive detection; higher -> lower CPU.
	"analysis_interval_s": 0.5,
	# Step 2.3: minimum revolutions (estimated) required to emit a jump event.
	"min_revs": 0.0,
}
_jump_config: Dict[str, float] = dict(JUMP_CONFIG_DEFAULTS)

# Async queue + worker task for decoupled jump analysis (Option A).
_jump_sample_queue: Optional[asyncio.Queue] = None
_jump_worker_task: Optional[asyncio.Task] = None
_frame_sync_task: Optional[asyncio.Task] = None

# Rolling IMU history for export / offline analysis.
IMU_HISTORY_MAX_SECONDS: float = 60.0
_imu_history: Deque[Dict[str, Any]] = deque()

# Rolling video frame history for real-time access and jump clip generation.
# Similar to _imu_history, but for video frames. Pruned based on time window.
FRAME_HISTORY_MAX_SECONDS: float = 120.0  # Keep 120 seconds (enough for multiple jumps + buffer)
_frame_history: Deque[Dict[str, Any]] = deque()

# In‑memory list of detected jump events for export / labelling.
_jump_events: Deque[Dict[str, Any]] = deque(maxlen=1000)
_next_event_id: int = 1

# In‑memory annotations keyed by event_id (name, note, future labels).
_jump_annotations: Dict[int, Dict[str, Any]] = {}

# Track jump windows per session to ensure uniqueness (avoid overlapping video clips)
# Format: {session_id: [(event_id, window_start, window_end), ...]}
_jump_windows_by_session: Dict[str, List[Tuple[int, float, float]]] = {}

# Manual gating for jump detection to avoid setup false‑positives.
_jump_detection_enabled: bool = False

# Lightweight debug counters (helps diagnose "no jumps detected" quickly).
_dbg: Dict[str, Any] = {
	"imu_packets": 0,
	"imu_samples": 0,
	"jump_queue_put_ok": 0,
	"jump_queue_put_drop": 0,
	"jump_worker_samples": 0,
	"jump_events_emitted": 0,
	"last_imu_t": None,
	"last_jump_t": None,
	"last_jump_event_id": None,
	"ble_payload_drop": 0,
	"ble_disconnects": 0,
	"last_ble_disconnect_t": None,
	"ble_payload_q_size": 0,
	"ble_payload_q_max": None,
	"loop_lag_ms_last": 0.0,
	"loop_lag_ms_max": 0.0,
	"collector_running": False,
	"collector_pid": None,
	"collector_last_pkt_t": None,
	"collector_disconnects": 0,
	"collector_last_disconnect_t": None,
	"collector_last_error": None,
	"imu_packets_5s": 0,
	"imu_samples_5s": 0,
	"imu_rate_hz_5s": 0.0,
	"collector_rx_packets_5s": None,
	"collector_rx_samples_5s": None,
	"collector_rx_rate_hz_5s": None,
	"collector_notify_stale_s": None,
	"clip_worker_pid": None,
	"clip_jobs_pending": 0,
	"video_proc_pid": None,
}

# Rolling receive-rate window (server-side ground truth, independent of browser timing)
_imu_rx_window: Deque[tuple[float, int]] = deque()  # (t_mono, n_samples)
_imu_rx_window_pkts: Deque[tuple[float, int]] = deque()  # (t_mono, 1)


def _jump_log_filter(message: str) -> None:
	"""
	Filter JumpDetector log lines before sending them to clients.

	- Always forward Phase 1/2 diagnostics.
	- Only forward [Jump] lines once detection has been manually enabled.
	"""
	global _jump_detection_enabled
	if "[Jump]" in message and not _jump_detection_enabled:
		return
	_log_to_clients(f"[JumpDetector] {message}")


@asynccontextmanager
async def lifespan(app: FastAPI):
	global _imu_proc, _imu_udp_transport, _jump_sample_queue, _jump_worker_task, _imu_history, _frame_history, _jump_events, _next_event_id, _jump_annotations, _clip_worker_proc, _video_proc
	try:
		# Initialise database (if configured).
		try:
			await db.init_db()
		except Exception as e:
			# DB is optional; log to clients and continue without persistence.
			print(f"[DB] init_db failed: {e!r}")
			# We deliberately don't call _log_to_clients here because WS manager
			# may not be ready yet.

		# Start decoupled jump‑analysis worker and IMU history.
		_jump_sample_queue = asyncio.Queue(maxsize=2000)
		_imu_history = deque()
		_frame_history = deque()
		state._jump_sample_queue = _jump_sample_queue
		state._frame_history = _frame_history
		_jump_events = deque(maxlen=1000)
		_next_event_id = 1
		_jump_annotations = {}
		_jump_worker_task = asyncio.create_task(_jump_worker_loop())
		# Start frame sync task to pull frames from backend to in-memory buffer.
		_frame_sync_task = asyncio.create_task(_frame_sync_loop())

		# Start UDP receiver for IMU packets from collector process.
		loop = asyncio.get_running_loop()

		class _ImuUdpProtocol(asyncio.DatagramProtocol):
			def datagram_received(self, data: bytes, addr):  # type: ignore[override]
				# Parse JSON (best-effort)
				try:
					msg = json.loads(data.decode("utf-8"))
				except Exception:
					return

				# Collector log passthrough
				if isinstance(msg, dict) and msg.get("type") == "log":
					txt = msg.get("msg")
					if isinstance(txt, str):
						_log_to_clients(txt)
					return

				# Collector telemetry passthrough
				if isinstance(msg, dict) and msg.get("type") == "collector_stat":
					try:
						if "disconnects" in msg:
							_dbg["collector_disconnects"] = int(msg.get("disconnects") or 0)
						if "last_disconnect_t" in msg:
							_dbg["collector_last_disconnect_t"] = msg.get("last_disconnect_t")
						if "last_error" in msg:
							_dbg["collector_last_error"] = msg.get("last_error")
						if "rx_samples_5s" in msg:
							_dbg["collector_rx_samples_5s"] = msg.get("rx_samples_5s")
						if "rx_packets_5s" in msg:
							_dbg["collector_rx_packets_5s"] = msg.get("rx_packets_5s")
						if "rx_rate_hz_5s" in msg:
							_dbg["collector_rx_rate_hz_5s"] = msg.get("rx_rate_hz_5s")
						if "notify_stale_s" in msg:
							_dbg["collector_notify_stale_s"] = msg.get("notify_stale_s")
					except Exception:
						pass
					return

				# IMU packet from collector
				if not isinstance(msg, dict) or msg.get("type") != "imu":
					return

				try:
					_dbg["collector_last_pkt_t"] = float(time.time())
				except Exception:
					pass

				# Collector-provided clock calibration (device timestamp -> epoch seconds)
				try:
					if isinstance(msg.get("imu_clock_offset_s"), (int, float)):
						_dbg["imu_clock_offset_s"] = float(msg.get("imu_clock_offset_s"))
				except Exception:
					pass
				try:
					if isinstance(msg.get("imu_clock_offset_fixed"), bool):
						_dbg["imu_clock_offset_fixed"] = bool(msg.get("imu_clock_offset_fixed"))
					if isinstance(msg.get("imu_clock_offset_calib_n"), (int, float)):
						_dbg["imu_clock_offset_calib_n"] = int(msg.get("imu_clock_offset_calib_n"))
				except Exception:
					pass

				# Update rate/mode if present (collector is authoritative once running)
				global _active_rate, _active_mode
				try:
					if isinstance(msg.get("rate"), (int, float)):
						_active_rate = int(msg.get("rate"))
					if isinstance(msg.get("mode"), str):
						_active_mode = str(msg.get("mode"))
				except Exception:
					pass

				samples = msg.get("samples") or []
				if not isinstance(samples, list) or not samples:
					return

				# Server-side rolling 5s receive rate (independent of browser)
				try:
					now_m = time.monotonic()
					_imu_rx_window.append((now_m, int(len(samples))))
					_imu_rx_window_pkts.append((now_m, 1))
					cut = now_m - 5.0
					while _imu_rx_window and _imu_rx_window[0][0] < cut:
						_imu_rx_window.popleft()
					while _imu_rx_window_pkts and _imu_rx_window_pkts[0][0] < cut:
						_imu_rx_window_pkts.popleft()
					s5 = sum(n for _, n in _imu_rx_window)
					p5 = len(_imu_rx_window_pkts)
					_dbg["imu_samples_5s"] = int(s5)
					_dbg["imu_packets_5s"] = int(p5)
					_dbg["imu_rate_hz_5s"] = float(s5) / 5.0
				except Exception:
					pass

				# Update debug counters based on collector payload.
				try:
					_dbg["imu_packets"] = int(_dbg.get("imu_packets", 0)) + 1
					_dbg["imu_samples"] = int(_dbg.get("imu_samples", 0)) + int(len(samples))
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
					if _jump_sample_queue is not None:
						try:
							_jump_sample_queue.put_nowait(
								{
									"t": t_i,
									"acc": s.get("acc", []),
									"gyro": s.get("gyro", []),
									"mag": s.get("mag", []),
									"imu_timestamp": s.get("imu_timestamp"),
									"imu_sample_index": s.get("imu_sample_index"),
								}
							)
							_dbg["jump_queue_put_ok"] = int(_dbg.get("jump_queue_put_ok", 0)) + 1
						except Exception:
							try:
								_dbg["jump_queue_put_drop"] = int(_dbg.get("jump_queue_put_drop", 0)) + 1
							except Exception:
								pass

					# History
					try:
						_imu_history.append(
							{
								"t": t_i,
								"imu_timestamp": s.get("imu_timestamp"),
								"imu_sample_index": s.get("imu_sample_index"),
								"acc": s.get("acc", []),
								"gyro": s.get("gyro", []),
								"mag": s.get("mag", []),
							}
						)
						if len(_imu_history) > 100:
							cutoff = t_i - IMU_HISTORY_MAX_SECONDS
							while _imu_history:
								first_t = float(_imu_history[0].get("t", 0.0))
								if first_t >= cutoff:
									break
								_imu_history.popleft()
					except Exception:
						pass

					# last_imu_t
					try:
						_dbg["last_imu_t"] = float(t_i)
						# Diagnostic: how far behind/ahead the incoming IMU timestamps are vs server wall time.
						_dbg["imu_now_minus_t_s"] = float(time.time()) - float(t_i)
					except Exception:
						pass

					# Recording IMU CSV
					try:
						if _imu_csv_fh is not None:
							acc = s.get("acc", []) or []
							gyro = s.get("gyro", []) or []
							mag = s.get("mag", []) or []
							ax, ay, az = (acc + [None, None, None])[:3]
							gx, gy, gz = (gyro + [None, None, None])[:3]
							mx, my, mz = (mag + [None, None, None])[:3]
							_imu_csv_fh.write(
								f"{t_i},{s.get('imu_timestamp','')},{s.get('imu_sample_index','')},"
								f"{ax if ax is not None else ''},{ay if ay is not None else ''},{az if az is not None else ''},"
								f"{gx if gx is not None else ''},{gy if gy is not None else ''},{gz if gz is not None else ''},"
								f"{mx if mx is not None else ''},{my if my is not None else ''},{mz if mz is not None else ''}\n"
							)
					except Exception:
						pass

				# Broadcast to WS clients (preserve existing UI message shape).
				try:
					asyncio.create_task(
						manager.broadcast_json(
							{
								"t": time.time(),
								"mode": msg.get("mode"),
								"rate": msg.get("rate"),
								"seq": msg.get("seq"),
								"timestamp": msg.get("timestamp"),
								"samples_len": len(samples),
								# Include per-sample host timestamp so the index page can render human-readable time.
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
			lambda: _ImuUdpProtocol(),
			local_addr=(IMU_UDP_HOST, IMU_UDP_PORT),
		)
		_imu_udp_transport = transport

		# Start jump clip worker process (file-queue based)
		try:
			jobs_dir = Path(JUMP_CLIP_JOBS_DIR)
			jobs_dir.mkdir(parents=True, exist_ok=True)
			cmd = [sys.executable, "-m", "modules.jump_clip_worker", "--jobs-dir", str(jobs_dir)]
			# Write worker logs to disk so clip failures are diagnosable on the Pi.
			log_path = jobs_dir / "worker.log"
			log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
			_clip_worker_proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
			_dbg["clip_worker_pid"] = int(_clip_worker_proc.pid) if _clip_worker_proc.pid else None
		except Exception:
			_clip_worker_proc = None
			_dbg["clip_worker_pid"] = None

		# Optionally start a dedicated video collector process (keeps camera/GPU code isolated).
		try:
			_video_proc = start_video_collector_subprocess(CFG)
			_dbg["video_proc_pid"] = int(_video_proc.pid) if _video_proc and _video_proc.pid else None
		except Exception:
			_video_proc = None
			_dbg["video_proc_pid"] = None

		yield
	finally:
		# Stop optional video collector process
		if _video_proc is not None:
			try:
				_video_proc.terminate()
			except Exception:
				pass
			try:
				_video_proc.wait(timeout=3)
			except Exception:
				pass
			_video_proc = None
			try:
				_dbg["video_proc_pid"] = None
			except Exception:
				pass

		# Stop jump clip worker process
		if _clip_worker_proc is not None:
			try:
				_clip_worker_proc.terminate()
			except Exception:
				pass
			try:
				_clip_worker_proc.wait(timeout=3)
			except Exception:
				pass
			_clip_worker_proc = None
			try:
				_dbg["clip_worker_pid"] = None
			except Exception:
				pass

		# Stop IMU collector process
		if _imu_proc is not None:
			try:
				_imu_proc.terminate()
			except Exception:
				pass
			try:
				_imu_proc.wait(timeout=3)
			except Exception:
				pass
			_imu_proc = None

		# Stop UDP receiver
		if _imu_udp_transport is not None:
			try:
				_imu_udp_transport.close()
			except Exception:
				pass
			_imu_udp_transport = None

		# Stop jump‑analysis worker
		if _jump_worker_task:
			_jump_worker_task.cancel()
			try:
				await _jump_worker_task
			except asyncio.CancelledError:
				pass
			except Exception:
				pass
		_jump_worker_task = None
		_jump_sample_queue = None

		# Stop frame sync task
		if _frame_sync_task:
			_frame_sync_task.cancel()
			try:
				await _frame_sync_task
			except asyncio.CancelledError:
				pass
			except Exception:
				pass
		_frame_sync_task = None


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
		asyncio.create_task(state.manager.broadcast_json({"type": "log", "msg": message}))
	except RuntimeError:
		pass


async def _frame_sync_loop() -> None:
	"""
	Periodic task that syncs frames from video backend to in-memory _frame_history buffer.
	Runs continuously during server lifetime. Similar pattern to IMU buffering.
	
	The backend accumulates frames during recording in its own buffer; this loop
	periodically copies them to the server's _frame_history for fast access by
	jump detection and clip generation code.
	"""
	global _frame_history, _video
	_last_sync_frame_idx: int = -1  # Track by frame index to avoid timestamp drift issues
	
	while True:
		try:
			await asyncio.sleep(0.1)  # Sync every 100ms (10 Hz)
			
			# Only sync if recording is active
			if state._session_id is None:
				# No active session; clear history if it's stale
				_last_sync_frame_idx = -1
				# Don't clear history immediately; it might be needed for recent jumps
				continue
			
			# Get all frames from backend (since frame 0)
			# Backend's get_frames_since filters by t_host, but we track by index to avoid gaps
			all_backend_frames = []
			try:
				if _video is not None:
					# Get frames from the start of current session (or all if not tracking session start)
					# We'll filter to only new ones using frame_idx
					all_backend_frames = _video.get_frames_since(0.0, include_frame_idx=True, include_width_height=True)
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
					_frame_history.append(f)
					_last_sync_frame_idx = max(_last_sync_frame_idx, frame_idx)
					new_count += 1
				except Exception:
					continue
			
			# Prune old frames (keep last FRAME_HISTORY_MAX_SECONDS)
			try:
				if _frame_history:
					cutoff = time.time() - FRAME_HISTORY_MAX_SECONDS
					# Since frames are added in order, we can prune from the front
					while _frame_history:
						first_t = float(_frame_history[0].get("t_host", 0.0))
						if first_t >= cutoff:
							break
						# If we prune, reset sync index to avoid gaps
						pruned_idx = int(_frame_history[0].get("frame_idx", -1))
						if pruned_idx >= 0:
							_last_sync_frame_idx = max(-1, pruned_idx - 1)
						_frame_history.popleft()
			except Exception:
				pass
				
		except asyncio.CancelledError:
			break
		except Exception:
			# Keep loop running even on errors
			await asyncio.sleep(0.5)


def _ensure_unique_jump_window(
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
	global _jump_windows_by_session
	
	if not session_id:
		# No session tracking, return base window
		return (base_window_start, base_window_end)
	
	# Get existing windows for this session
	existing_windows = _jump_windows_by_session.get(session_id, [])
	
	window_start = base_window_start
	window_end = base_window_end
	
	# Check for overlaps and adjust
	overlap_tolerance = 0.1  # Small tolerance to avoid exact boundary overlaps
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
			min_window_size = 2.0  # Minimum 2 seconds
			if window_end - window_start < min_window_size:
				# If adjustment made window too small, center it on t_peak
				window_center = t_peak
				window_start = window_center - min_window_size / 2.0
				window_end = window_center + min_window_size / 2.0
			
			# Ensure window_start < window_end
			if window_start >= window_end:
				# Fallback: use a tight window around t_peak
				window_start = t_peak - 1.0
				window_end = t_peak + 1.0
	
	# Store this window for future overlap checks
	if session_id not in _jump_windows_by_session:
		_jump_windows_by_session[session_id] = []
	_jump_windows_by_session[session_id].append((event_id, window_start, window_end))
	
	# Clean up old windows (keep last 100 per session to avoid unbounded growth)
	if len(_jump_windows_by_session[session_id]) > 100:
		_jump_windows_by_session[session_id] = _jump_windows_by_session[session_id][-100:]
	
	return (window_start, window_end)


async def _jump_worker_loop() -> None:
	"""
	Background task that consumes IMU samples from the queue and runs
	JumpDetectorRealtime to emit structured jump events, decoupled from BLE.
	"""
	global _jump_sample_queue, _jump_events, _next_event_id, _jump_annotations

	# One detector instance per worker, recreated when stream rate changes.
	jump_detector: Optional[JumpDetectorRealtime] = None
	last_rate: Optional[int] = None

	while True:
		if _jump_sample_queue is None:
			# Should not happen often; be defensive.
			await asyncio.sleep(0.1)
			continue

		sample = await _jump_sample_queue.get()
		try:
			_dbg["jump_worker_samples"] = int(_dbg.get("jump_worker_samples", 0)) + 1
		except Exception:
			pass
		# Lazily create (or recreate) detector with the active stream rate.
		global _active_rate
		active_rate = int(_active_rate) if int(_active_rate) > 0 else RATE
		if jump_detector is None or last_rate != active_rate:
			jump_detector = JumpDetectorRealtime(
				sample_rate_hz=active_rate,
				window_seconds=3.0,
				logger=_jump_log_filter,
				config=_jump_config,
			)
			last_rate = active_rate
		try:
			events = (jump_detector.update(sample) if jump_detector else []) or []
		except Exception:
			events = []

		if not events:
			continue

		# Only emit jump messages once detection has been explicitly enabled.
		if not _jump_detection_enabled:
			continue

		for ev in events:
			try:
				event_id = _next_event_id
				_next_event_id += 1
				try:
					_dbg["jump_events_emitted"] = int(_dbg.get("jump_events_emitted", 0)) + 1
					_dbg["last_jump_t"] = float(ev.get("t_peak") or time.time())
					_dbg["last_jump_event_id"] = int(event_id)
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
					_dbg["jump_emit_delay_s"] = float(time.time()) - float(t_peak_val)
					if ev.get("t_takeoff") is not None:
						_dbg["jump_emit_delay_from_takeoff_s"] = float(time.time()) - float(ev.get("t_takeoff"))  # type: ignore[arg-type]
				except Exception:
					pass

				# Keep a compact record for export / labelling.
				record = dict(jump_msg)
				record["t_takeoff"] = ev.get("t_takeoff")
				record["t_landing"] = ev.get("t_landing")
				_jump_events.append(record)

				# Ensure there is at least a stub annotation (name can be overridden later).
				if event_id not in _jump_annotations:
					_jump_annotations[event_id] = {"name": f"Jump {event_id}", "note": None}
				asyncio.create_task(manager.broadcast_json(jump_msg))

				# Fire‑and‑forget persistence of this jump and its IMU window to DB.
				try:
					# Use configurable window around t_peak for DB storage.
					t_peak_val = float(ev.get("t_peak", 0.0))
					t_takeoff_val = float(ev.get("t_takeoff", t_peak_val))
					t_landing_val = float(ev.get("t_landing", t_peak_val))
					pre_jump_s = float(CFG.jump_recording.pre_jump_seconds)
					post_jump_s = float(CFG.jump_recording.post_jump_seconds)
					# Window should be centered on t_peak, but ensure it includes takeoff/landing
					base_window_start = min(t_takeoff_val - 0.5, t_peak_val - pre_jump_s)
					base_window_end = max(t_landing_val + 0.5, t_peak_val + post_jump_s)
					
					# Ensure window uniqueness: check for overlaps with existing jumps in same session
					# and adjust boundaries to avoid overlap
					window_start, window_end = _ensure_unique_jump_window(
						state._session_id,
						event_id,
						base_window_start,
						base_window_end,
						t_peak_val,
					)
					ann = _jump_annotations.get(event_id) or {}
					jump_for_db = dict(record)
					# Link to current recording session (if any) so jumps can be played back.
					jump_for_db["session_id"] = state._session_id
					jump_for_db["t_start"] = window_start
					jump_for_db["t_end"] = window_end
					# Persist immediately on detection. Any heavy post-processing (ffmpeg clip cutting,
					# jump_frames generation) is offloaded to a separate worker process via job file.
					#
					# IMPORTANT: bind values as explicit function arguments.
					# This avoids a classic late-binding closure bug where multiple create_task(...)
					# calls inside the loop would all see the "last" event_id/jump_for_db/window values.
					async def _persist_and_enqueue(
						jump_for_db_arg: Dict[str, Any],
						ann_arg: Dict[str, Any],
						window_start_arg: float,
						window_end_arg: float,
						event_id_arg: int,
						t_peak_val_arg: float,
					) -> None:
						# Critical: at t_peak time we usually DO NOT yet have the future IMU samples.
						# Wait briefly for the stream to advance so we can persist the full requested window.
						# Optimized: reduce wait time since we have in-memory buffer, and use more aggressive polling.
						try:
							post_jump_s = float(CFG.jump_recording.post_jump_seconds)
							deadline = time.time() + post_jump_s + 0.5  # Reduced from 1.5s to 0.5s slack
							wait_start = time.time()
							while time.time() < deadline:
								last_t = _dbg.get("last_imu_t")
								try:
									last_t_f = float(last_t) if last_t is not None else None
								except Exception:
									last_t_f = None
								if last_t_f is not None and last_t_f >= float(window_end_arg):
									break
								await asyncio.sleep(0.02)  # More aggressive polling: 20ms instead of 50ms
							try:
								_dbg["db_window_waited_s"] = float(max(0.0, time.time() - wait_start))
							except Exception:
								pass
						except Exception:
							pass

						# Slice history AFTER waiting, so we include samples up through window_end.
						window_samples = []
						try:
							history_list = list(_imu_history) if _imu_history else []
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
								_dbg["db_last_insert_error"] = "insert_jump_with_imu returned no jump_id (DB disabled or insert skipped)"
								return
							_dbg["db_last_insert_ok_t"] = float(time.time())
							_dbg["db_last_insert_error"] = None
							_log_to_clients(
								f"[DB] Inserted jump: event_id={event_id_arg}, jump_id={jump_id}, samples={len(window_samples)} "
								f"(window={window_start_arg:.3f}->{window_end_arg:.3f})"
							)
						except Exception as e:
							_dbg["db_last_insert_error"] = repr(e)
							_log_to_clients(f"[DB] insert_jump_with_imu failed for event_id={event_id_arg}: {e!r}")
							print(f"[DB] insert_jump_with_imu failed for event_id={event_id_arg}: {e!r}")
							return

						# Enqueue clip generation job. This runs out-of-process and may complete later
						# (e.g., after MP4 mux is available).
						try:
							clip_buffer_s = float(CFG.jump_recording.clip_buffer_seconds)
							clip_start_host = float(window_start_arg) - clip_buffer_s
							clip_end_host = float(window_end_arg) + clip_buffer_s
							clip_duration = clip_end_host - clip_start_host
							# Debug logging for clip length issues
							_log_to_clients(
								f"[Clip] event_id={event_id_arg}: window=[{window_start_arg:.3f}, {window_end_arg:.3f}], "
								f"clip=[{clip_start_host:.3f}, {clip_end_host:.3f}], duration={clip_duration:.2f}s"
							)
						except Exception:
							pre_jump_s = float(CFG.jump_recording.pre_jump_seconds)
							post_jump_s = float(CFG.jump_recording.post_jump_seconds)
							clip_start_host = float(t_peak_val_arg) - pre_jump_s - 0.4
							clip_end_host = float(t_peak_val_arg) + post_jump_s + 0.4

						_enqueue_jump_clip_job(
							{
								"jump_id": int(jump_id),
								"event_id": int(event_id_arg),
								"session_id": str(state._session_id or ""),
								"clip_start_host": float(clip_start_host),
								"clip_end_host": float(clip_end_host),
								"video_fps": 30,
								"wait_mp4_timeout_s": 900,
							}
						)

					try:
						ws_ = float(window_start)
						we_ = float(window_end)
					except Exception:
						pre_jump_s = float(CFG.jump_recording.pre_jump_seconds)
						post_jump_s = float(CFG.jump_recording.post_jump_seconds)
						ws_ = float(t_peak_val) - pre_jump_s
						we_ = float(t_peak_val) + post_jump_s
					try:
						tp_ = float(t_peak_val)
					except Exception:
						tp_ = time.time()
					asyncio.create_task(
						_persist_and_enqueue(
							dict(jump_for_db),
							dict(ann),
							ws_,
							we_,
							int(event_id),
							tp_,
						)
					)
				except Exception as e:
					# DB persistence is best‑effort; ignore errors here.
					print(f"[DB] Error scheduling insert_jump_with_imu: {e!r}")
			except Exception:
				# Jump notifications are best‑effort; ignore errors.
				pass
def _session_base_dir(session_id: str) -> Path:
	# Session directories are stored under <sessions.base_dir>/<session_id>/
	# NOTE: Do not call get_config() here because server.py defines an async route handler
	# named get_config(), which can shadow the imported modules.config.get_config symbol.
	# Use the already-loaded global CFG instead.
	base = Path(CFG.sessions.base_dir or str(Path("data") / "sessions"))
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


# Populate shared state for routers (must run after all referenced defs: _session_base_dir, _log_to_clients, _enqueue_jump_clip_job, etc.)
state.manager = manager
state.get_page_html = get_page_html
state.UI_DIR = UI_DIR
state._video = _video
state._session_id = _session_id
state._session_dir = _session_dir
state._session_lock = _session_lock
state._frame_history = _frame_history
state._dbg = _dbg
state.CFG = CFG
state._active_mode = _active_mode
state._active_rate = _active_rate
state._jump_config = _jump_config
state._jump_detection_enabled = _jump_detection_enabled
state._session_base_dir = _session_base_dir
state._log_to_clients = _log_to_clients
state._run_pose_for_jump_best_effort = _run_pose_for_jump_best_effort
state._maybe_schedule_pose_for_jump = _maybe_schedule_pose_for_jump
state._enqueue_jump_clip_job = _enqueue_jump_clip_job
state._count_clip_jobs_pending = _count_clip_jobs_pending
state._count_clip_jobs_done = _count_clip_jobs_done
state._count_clip_jobs_failed = _count_clip_jobs_failed
state._read_last_clip_job_error = _read_last_clip_job_error

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
		rows = await db.list_jumps(limit=2000)
		jumps_in_session = [r for r in rows if r.get("session_id") == session_id]
		if not jumps_in_session:
			return

		clip_backend = get_video_backend_for_tools()

		clips_dir = base / (CFG.sessions.jump_clips_subdir or "jump_clips")
		clips_dir.mkdir(parents=True, exist_ok=True)

		# Load session frames (from DB if present, otherwise from file endpoint fallback).
		# We'll use these to store per-jump clip-relative frame timing linked to each jump.
		session_frames: list[dict] = []
		try:
			session_frames = await db.get_frames(session_id=session_id, limit=500000)
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
				pre = 0.8
				post = 0.8
				if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and float(t1) > float(t0):
					clip_start_host = float(t0) - pre
					clip_end_host = float(t1) + post
				elif isinstance(tp, (int, float)):
					clip_start_host = float(tp) - 1.2
					clip_end_host = float(tp) + 1.2
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
					rel = str(Path(CFG.sessions.base_dir or str(Path("data") / "sessions")) / session_id / (CFG.sessions.jump_clips_subdir or "jump_clips") / out_name)
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
async def connect_device(payload: ConnectPayload):
	"""
	Start the IMU collector process connected to the specified device (MAC or name).
	The collector streams IMU packets to this server over localhost UDP.
	If a collector is already running, it will be stopped and replaced.

	Can accept either:
	- "device": MAC address or device name
	- "skater_id": skater ID (will use first registered device for that skater)
	"""
	global _imu_proc, _active_rate, _active_mode

	try:
		device = (payload.device or "").strip()
		skater_id = payload.skater_id
		mode = (payload.mode or MODE).strip().upper()
		rate = int(payload.rate or RATE)
		# Persist the active stream parameters for the analysis worker.
		_active_rate = rate
		_active_mode = mode

		# If skater_id is provided, get the first device for that skater
		if skater_id:
			skater_devices = await db.get_skater_devices(int(skater_id))
			if not skater_devices:
				raise HTTPException(status_code=400, detail="No sensor registered for this skater")
			# Use the first device (prefer waist placement if available)
			waist_device = next((d for d in skater_devices if d.get("placement") == "waist"), None)
			selected_device = waist_device or skater_devices[0]
			device = selected_device.get("mac_address")
			if not device:
				raise HTTPException(status_code=400, detail="Device MAC address not found")
		elif not device:
			raise HTTPException(status_code=400, detail="Missing 'device' or 'skater_id' in request body.")

		# Resolve device name to MAC address if it's a registered device name
		resolved_mac = await db.resolve_device_identifier(device)
		if resolved_mac:
			device = resolved_mac  # Use MAC address for connection

		# Stop existing collector
		if _imu_proc is not None:
			try:
				_imu_proc.terminate()
			except Exception:
				pass
			try:
				_imu_proc.wait(timeout=3)
			except Exception:
				pass
			_imu_proc = None

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
		_imu_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		try:
			_dbg["collector_running"] = True
			_dbg["collector_pid"] = int(_imu_proc.pid) if _imu_proc.pid else None
		except Exception:
			pass

		_log_to_clients(f"[Collector] Started PID={_imu_proc.pid} (mode={mode}, rate={rate}, udp={IMU_UDP_HOST}:{IMU_UDP_PORT})")
		return {"detail": f"Starting IMU collector for {device} (mode={mode}, rate={rate})", "collector_pid": _imu_proc.pid}
	except HTTPException:
		# Pass through explicit HTTP errors
		raise
	except Exception as e:
		_log_to_clients(f"/connect error: {e!r}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/disconnect")
async def disconnect_device():
	"""
	Stop the current IMU collector process (if any).
	"""
	global _imu_proc

	if _imu_proc is not None:
		try:
			_imu_proc.terminate()
		except Exception:
			pass
		try:
			_imu_proc.wait(timeout=3)
		except Exception:
			pass
		_imu_proc = None

		try:
			_dbg["collector_running"] = False
			_dbg["collector_pid"] = None
		except Exception:
			pass

	_log_to_clients("IMU collector stopped via /disconnect")
	return {"detail": "Stopped IMU collector"}


@app.post("/detection/start")
async def start_jump_detection():
	"""
	Manually enable jump detection, to avoid false positives during sensor
	setup / strapping on.
	"""
	global _jump_detection_enabled, _detection_session_id
	_jump_detection_enabled = True
	_log_to_clients("[JumpDetector] Detection ENABLED via /detection/start")

	# New simplified workflow:
	# When detection starts, also start a recording session (best-effort) so every
	# detected jump can later be paired with a video clip.
	if _detection_session_id is None:
		sid = time.strftime("%Y%m%d_%H%M%S") + "_detect"
		try:
			# Only set _detection_session_id if session_start succeeds.
			await session_start({"session_id": sid})
			_detection_session_id = sid
			_log_to_clients(f"[Session] Auto-started recording session for detection: {sid}")
		except HTTPException as e:
			_detection_session_id = None
			_log_to_clients(f"[Session] Auto-start for detection failed: {e.detail}")
			print(f"[Session] Auto-start for detection failed: {e.detail}")
			# Surface the error to the caller/UI so it's not silently ignored.
			raise
		except Exception as e:
			_detection_session_id = None
			_log_to_clients(f"[Session] Auto-start for detection failed: {e!r}")
			print(f"[Session] Auto-start for detection failed: {e!r}")
			raise HTTPException(status_code=500, detail=f"Auto-start recording failed: {e!r}")

	return {"detail": "Jump detection enabled (recording auto-started).", "session_id": _detection_session_id}


@app.post("/detection/stop")
async def stop_jump_detection():
	"""
	Manually disable jump detection while keeping BLE streaming active.
	"""
	global _jump_detection_enabled, _detection_session_id
	_jump_detection_enabled = False
	_log_to_clients("[JumpDetector] Detection DISABLED via /detection/stop")

	# Stop auto-started session (best-effort)
	try:
		if _detection_session_id is not None:
			await session_stop()
			_log_to_clients(f"[Session] Auto-stopped recording session for detection: {_detection_session_id}")
	finally:
		_detection_session_id = None

	return {"detail": "Jump detection disabled (recording auto-stopped)."}


@app.get("/annotations/{event_id}")
async def get_annotation(event_id: int):
	"""
	Fetch stored annotation (name/note, later labels) for a jump event.
	Returns {} if none exists.
	"""
	return _jump_annotations.get(event_id, {})


@app.post("/annotations/{event_id}")
async def set_annotation(event_id: int, payload: Dict[str, Any]):
	"""
	Store or update annotation for a jump event (name, note, future label fields).
	"""
	global _jump_annotations
	data = payload or {}
	current = _jump_annotations.get(event_id, {})
	current.update(
		{
			"name": data.get("name", current.get("name")),
			"note": data.get("note", current.get("note")),
		}
	)
	_jump_annotations[event_id] = current
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
async def get_config():
	"""
	Return current jump detection configuration.
	"""
	return {"jump": _jump_config}


@app.post("/config")
async def set_config(payload: Dict[str, Any]):
	"""
	Update jump detection configuration. Changes take effect on the next
	connection (when a new JumpDetectorRealtime is created).
	DEPRECATED: Use /api/skaters/{skater_id}/detection-settings instead.
	"""
	global _jump_config

	body = payload or {}
	jump_cfg = body.get("jump") or {}
	if not isinstance(jump_cfg, dict):
		raise HTTPException(status_code=400, detail="Field 'jump' must be an object")

	for key, value in jump_cfg.items():
		if key in JUMP_CONFIG_DEFAULTS:
			try:
				_jump_config[key] = float(value)
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
async def export_imu(seconds: float = 30.0, mode: str = "raw"):
	"""
	Export recent IMU data for offline analysis / labelling.

	Modes:
	  - mode="raw" (default): return IMU samples from the last `seconds`.
	  - mode="jumps": return per‑jump windows keyed by event_id, using
	    detected jump events within the last `seconds`.
	"""
	now = time.time()
	horizon = max(0.0, float(seconds))
	cutoff = now - horizon

	history_snapshot = list(_imu_history)

	if mode == "jumps":
		# Export per‑jump windows keyed by event_id. We use t_takeoff/t_landing
		# if available; otherwise fall back to a symmetric window around t_peak.
		events_snapshot = list(_jump_events)
		windows = []
		for ev in events_snapshot:
			t_peak = float(ev.get("t_peak", 0.0))
			if t_peak < cutoff:
				continue
			t0 = ev.get("t_takeoff")
			t1 = ev.get("t_landing")
			if t0 is None or t1 is None:
				half = 0.75
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

			ann = _jump_annotations.get(int(ev.get("event_id", 0))) or {}

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


async def ble_worker(device: Optional[str], mode: str, rate: int) -> None:
	"""
	Deprecated: we now run BLE acquisition in a separate process (`modules.imu_collector`)
	and ingest IMU samples over localhost UDP in the server.

	This function is kept as a stub to avoid breaking older imports/scripts.
	"""
	raise RuntimeError("ble_worker is deprecated; use the IMU collector process.")