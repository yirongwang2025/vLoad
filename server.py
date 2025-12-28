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
from typing import Deque, Set, Optional, Dict, Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse

# Reuse your BLE client
from modules.movesense_gatt import MovesenseGATTClient
from modules import db
from modules.jump_detector import JumpDetectorRealtime
from modules.oakd_stream import OakdStreamManager

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

DEVICE = os.getenv("MOVE_DEVICE")  # e.g., "74:92:BA:10:F9:00" or exact name
MODE = os.getenv("MOVE_MODE", "IMU9")  # "IMU6" or "IMU9"
RATE = int(os.getenv("MOVE_RATE", "104"))

# IMU collector process globals (separate process architecture)
IMU_UDP_HOST = os.getenv("IMU_UDP_HOST", "127.0.0.1")
IMU_UDP_PORT = int(os.getenv("IMU_UDP_PORT", "9999"))
_imu_proc: Optional[subprocess.Popen] = None
_imu_udp_transport: Optional[asyncio.DatagramTransport] = None

# Jump clip worker process (offloads ffmpeg + heavy DB work from realtime detection)
JUMP_CLIP_JOBS_DIR = os.getenv("JUMP_CLIP_JOBS_DIR", str(Path("data") / "jobs" / "jump_clips"))
_clip_worker_proc: Optional[subprocess.Popen] = None

# Active stream settings (set on /connect). Used by analysis worker.
_active_mode: str = MODE
_active_rate: int = RATE

# OAK-D (DepthAI) camera manager (optional; only runs if user connects it)
_oakd = OakdStreamManager()

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
	# Step 2.3: minimum revolutions (estimated) required to emit a jump event.
	"min_revs": 0.0,
}
_jump_config: Dict[str, float] = dict(JUMP_CONFIG_DEFAULTS)

# Async queue + worker task for decoupled jump analysis (Option A).
_jump_sample_queue: Optional[asyncio.Queue] = None
_jump_worker_task: Optional[asyncio.Task] = None

# Rolling IMU history for export / offline analysis.
IMU_HISTORY_MAX_SECONDS: float = 60.0
_imu_history: Deque[Dict[str, Any]] = deque()

# In‑memory list of detected jump events for export / labelling.
_jump_events: Deque[Dict[str, Any]] = deque(maxlen=1000)
_next_event_id: int = 1

# In‑memory annotations keyed by event_id (name, note, future labels).
_jump_annotations: Dict[int, Dict[str, Any]] = {}

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
	global _imu_proc, _imu_udp_transport, _jump_sample_queue, _jump_worker_task, _imu_history, _jump_events, _next_event_id, _jump_annotations, _clip_worker_proc
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
		_jump_events = deque(maxlen=1000)
		_next_event_id = 1
		_jump_annotations = {}
		_jump_worker_task = asyncio.create_task(_jump_worker_loop())

		# Start UDP receiver for IMU packets from collector process.
		loop = asyncio.get_running_loop()

		class _ImuUdpProtocol(asyncio.DatagramProtocol):
			def datagram_received(self, data: bytes, addr):  # type: ignore[override]
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
						# Collector-side rate (BLE ground truth)
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

				if not isinstance(msg, dict) or msg.get("type") != "imu":
					return

				try:
					_dbg["collector_last_pkt_t"] = float(time.time())
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
				for s in samples:
					if not isinstance(s, dict):
						continue
					t_i = s.get("t")
					try:
						t_i = float(t_i)
					except Exception:
						t_i = None
					if t_i is None:
						continue

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
							try:
								_dbg["jump_queue_put_ok"] = int(_dbg.get("jump_queue_put_ok", 0)) + 1
							except Exception:
								pass
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
								"samples": [{"acc": s.get("acc", []), "gyro": s.get("gyro", []), "mag": s.get("mag", [])} for s in samples],
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
			_clip_worker_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			_dbg["clip_worker_pid"] = int(_clip_worker_proc.pid) if _clip_worker_proc.pid else None
		except Exception:
			_clip_worker_proc = None
			_dbg["clip_worker_pid"] = None

		yield
	finally:
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


app = FastAPI(lifespan=lifespan)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Load HTML templates from UI directory
INDEX_HTML = load_html_template("index.html")

JUMPS_HTML = load_html_template("jumps.html")

@app.get("/", response_class=HTMLResponse)
async def index():
	return INDEX_HTML


@app.get("/jumps", response_class=HTMLResponse)
async def jumps_page():
	return JUMPS_HTML

class ConnectionManager:
	def __init__(self) -> None:
		self._clients: Set[WebSocket] = set()
		self._lock = asyncio.Lock()

	async def connect(self, websocket: WebSocket) -> None:
		await websocket.accept()
		async with self._lock:
			self._clients.add(websocket)

	async def disconnect(self, websocket: WebSocket) -> None:
		async with self._lock:
			self._clients.discard(websocket)

	async def broadcast_json(self, message: Dict[str, Any]) -> None:
		payload = json.dumps(message, separators=(",", ":"))
		async with self._lock:
			if not self._clients:
				return
			send_tasks = []
			for ws in list(self._clients):
				send_tasks.append(self._send(ws, payload))
			await asyncio.gather(*send_tasks, return_exceptions=True)

	@staticmethod
	async def _send(ws: WebSocket, payload: str) -> None:
		try:
			await ws.send_text(payload)
		except Exception:
			try:
				await ws.close()
			except Exception:
				pass

manager = ConnectionManager()


def _log_to_clients(message: str) -> None:
	"""
	Send a log line to all connected WebSocket clients.
	Fire-and-forget; safe to call from non-async code.
	"""
	try:
		asyncio.create_task(manager.broadcast_json({"type": "log", "msg": message}))
	except RuntimeError:
		# No running loop yet; ignore
		pass


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
					# Use a slightly wider window around t_peak for DB storage: [-2s, +3s].
					t_peak_val = float(ev.get("t_peak", 0.0))
					window_start = t_peak_val - 2.0
					window_end = t_peak_val + 3.0
					# Optimized: use binary search to find window bounds without full conversion
					if not _imu_history:
						window_samples = []
					else:
						history_list = list(_imu_history)
						if not history_list:
							window_samples = []
						else:
							times = [float(row.get("t", 0.0)) for row in history_list]
							left_idx = bisect.bisect_left(times, window_start)
							right_idx = bisect.bisect_right(times, window_end)
							window_samples = history_list[left_idx:right_idx]
					ann = _jump_annotations.get(event_id) or {}
					jump_for_db = dict(record)
					# Link to current recording session (if any) so jumps can be played back.
					jump_for_db["session_id"] = _session_id
					jump_for_db["t_start"] = window_start
					jump_for_db["t_end"] = window_end
					# Persist immediately on detection. Any heavy post-processing (ffmpeg clip cutting,
					# jump_frames generation) is offloaded to a separate worker process via job file.
					async def _persist_and_enqueue() -> None:
						jump_id = await db.insert_jump_with_imu(jump_for_db, ann, window_samples)
						if not jump_id:
							return

						# Enqueue clip generation job. This runs out-of-process and may complete later
						# (e.g., after MP4 mux is available).
						try:
							clip_start_host = float(window_start) - 0.8
							clip_end_host = float(window_end) + 0.8
						except Exception:
							clip_start_host = float(t_peak_val) - 1.2
							clip_end_host = float(t_peak_val) + 1.2

						_enqueue_jump_clip_job(
							{
								"jump_id": int(jump_id),
								"event_id": int(event_id),
								"session_id": str(_session_id or ""),
								"clip_start_host": float(clip_start_host),
								"clip_end_host": float(clip_end_host),
								"video_fps": 30,
								"wait_mp4_timeout_s": 900,
							}
						)

					asyncio.create_task(_persist_and_enqueue())
				except Exception as e:
					# DB persistence is best‑effort; ignore errors here.
					print(f"[DB] Error scheduling insert_jump_with_imu: {e!r}")
			except Exception:
				# Jump notifications are best‑effort; ignore errors.
				pass
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
	await manager.connect(websocket)
	try:
		while True:
			# Keep connection alive; client doesn't need to send anything
			await websocket.receive_text()
	except WebSocketDisconnect:
		pass
	except Exception:
		pass
	finally:
		await manager.disconnect(websocket)


@app.get("/scan")
async def scan_devices():
	"""
	Scan for nearby Movesense devices and return a list of (address, name).
	"""
	devices = [
		{"address": addr, "name": name}
		async for addr, name in MovesenseGATTClient.scan_for_movesense(timeout=7.0)
	]
	return {"devices": devices}


@app.post("/video/connect")
async def video_connect():
	"""
	Connect to the OAK-D camera (DepthAI) and start MJPEG streaming.
	"""
	try:
		# Practical warning: USB3 devices (like OAK‑D) can introduce 2.4GHz interference
		# and cause BLE drops. This is usually hardware/RF, not Python load.
		try:
			if _dbg.get("collector_running") and int(_active_rate or 0) >= 104:
				_log_to_clients(
					"[Hint] Video starting while BLE IMU is at 104Hz+. If BLE drops, try IMU6/52Hz, move OAK‑D to a USB extension/ferrite, "
					"use 5GHz Wi‑Fi, or use an external BT dongle on an extension away from USB3."
				)
		except Exception:
			pass
		_oakd.start()
		# Give the background thread a moment to initialize and surface errors.
		await asyncio.sleep(0.2)
		st = _oakd.get_status()
		if not st.get("running") and st.get("error"):
			raise HTTPException(status_code=500, detail=str(st.get("error")))
		return {"detail": "OAK-D video streaming started.", "status": st}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Video connect failed: {e!r}")


@app.post("/video/disconnect")
async def video_disconnect():
	"""
	Stop the OAK-D camera stream.
	"""
	try:
		_oakd.stop()
		return {"detail": "OAK-D video streaming stopped.", "status": _oakd.get_status()}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Video disconnect failed: {e!r}")


@app.get("/video/status")
async def video_status():
	return _oakd.get_status()


@app.get("/video/mjpeg")
async def video_mjpeg(fps: float = 15.0):
	"""
	Live MJPEG stream from OAK-D (on-device MJPEG encoding).
	Browser can display via <img src="/video/mjpeg">.
	"""

	async def gen():
		boundary = b"frame"
		last_t = None
		last_sent_mono = 0.0
		try:
			max_fps = float(fps)
		except Exception:
			max_fps = 15.0
		if not (max_fps > 0.0):
			max_fps = 15.0
		min_interval = 1.0 / max_fps
		while True:
			jpeg, t = _oakd.get_latest_jpeg()
			if jpeg is None or t is None:
				await asyncio.sleep(0.05)
				continue
			# Avoid re-sending the same frame in a tight loop
			if last_t is not None and t == last_t:
				await asyncio.sleep(0.01)
				continue

			# Cap send rate (drop frames if producer is faster than max_fps)
			now_mono = time.monotonic()
			elapsed = now_mono - last_sent_mono
			if elapsed < min_interval:
				await asyncio.sleep(min_interval - elapsed)
				# Don't send a potentially stale frame; loop and grab the latest.
				continue

			last_t = t
			last_sent_mono = time.monotonic()
			yield b"--" + boundary + b"\r\n"
			yield b"Content-Type: image/jpeg\r\n"
			yield b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
			yield jpeg + b"\r\n"

	return StreamingResponse(
		gen(),
		media_type="multipart/x-mixed-replace; boundary=frame",
		headers={
			"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
			"Pragma": "no-cache",
			"Connection": "keep-alive",
		},
	)


@app.get("/video/snapshot.jpg")
async def video_snapshot():
	"""
	Return a single latest JPEG frame (useful to debug preview issues).
	"""
	jpeg, _t = _oakd.get_latest_jpeg()
	if jpeg is None:
		raise HTTPException(status_code=404, detail="No JPEG frame available yet")
	return Response(
		content=jpeg,
		media_type="image/jpeg",
		headers={
			"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
			"Pragma": "no-cache",
		},
	)


@app.get("/video/debug")
async def video_debug():
	"""
	Debug info about the latest encoded MJPEG packet (valid JPEG should start with FFD8 and end with FFD9).
	"""
	jpeg, t = _oakd.get_latest_jpeg()
	st = _oakd.get_status()
	if jpeg is None:
		return {"status": st, "has_jpeg": False}
	head = jpeg[:8].hex()
	tail = jpeg[-8:].hex() if len(jpeg) >= 8 else jpeg.hex()
	return {
		"status": st,
		"has_jpeg": True,
		"len": len(jpeg),
		"head_hex": head,
		"tail_hex": tail,
		"starts_ffd8": bool(len(jpeg) >= 2 and jpeg[0] == 0xFF and jpeg[1] == 0xD8),
		"ends_ffd9": bool(len(jpeg) >= 2 and jpeg[-2] == 0xFF and jpeg[-1] == 0xD9),
		"t_host": t,
	}


@app.post("/session/start")
async def session_start(payload: Dict[str, Any]):
	"""
	Start a recording session. Creates a session directory and enables:
	- OAK-D H264 recording + frames.csv
	- IMU sample logging to imu.csv (from BLE worker path)
	"""
	global _session_id, _session_dir, _imu_csv_fh
	async with _session_lock:
		if _session_id is not None:
			return {"detail": "Session already running", "session_id": _session_id}

		# Create session id
		sid = (payload.get("session_id") or "").strip()
		if not sid:
			sid = time.strftime("%Y%m%d_%H%M%S")

		base = Path("data") / "sessions" / sid
		base.mkdir(parents=True, exist_ok=True)

		# Save config snapshot
		t_start = time.time()
		try:
			(base / "session.json").write_text(
				json.dumps(
					{
						"session_id": sid,
						"t_start": t_start,
						"imu": {"mode": _active_mode, "rate": _active_rate},
						"jump_config": _jump_config,
					},
					indent=2,
				),
				encoding="utf-8",
			)
		except Exception:
			pass

		# Persist session metadata to DB (best-effort)
		try:
			await db.upsert_session_start(
				session_id=sid,
				t_start=t_start,
				imu_mode=_active_mode,
				imu_rate=_active_rate,
				jump_config=_jump_config,
				video_fps=30,
				video_path=str(Path("data") / "sessions" / sid / "video.mp4"),
				meta=None,
			)
		except Exception:
			pass

		# Enable OAK-D recording (ensure camera is running)
		_oakd.start()
		_oakd.start_recording(str(base), fps=30)

		# Open IMU CSV
		imu_path = base / "imu.csv"
		imu_new = not imu_path.exists()
		_imu_csv_fh = open(imu_path, "a", encoding="utf-8", newline="\n")
		if imu_new:
			_imu_csv_fh.write("t,imu_timestamp,imu_sample_index,ax,ay,az,gx,gy,gz,mx,my,mz\n")
			_imu_csv_fh.flush()

		_session_id = sid
		_session_dir = base
		return {"detail": "Session started", "session_id": sid, "dir": str(base)}


@app.post("/session/stop")
async def session_stop():
	global _session_id, _session_dir, _imu_csv_fh
	async with _session_lock:
		if _session_id is None:
			return {"detail": "No active session"}

		sid = _session_id
		# Stop video recording (keep preview running)
		try:
			_oakd.stop_recording()
		except Exception:
			pass

		# Best-effort: if ffmpeg is available, mux the raw H264 bitstream into MP4
		# so the browser can play it back.
		try:
			import shutil
			import subprocess

			if _session_dir is not None:
				h264 = _session_dir / "video.h264"
				mp4 = _session_dir / "video.mp4"
				if h264.exists() and (not mp4.exists()):
					ffmpeg = shutil.which("ffmpeg")
					if ffmpeg:
						# Use fixed FPS (recording is 30fps today) and generate PTS.
						subprocess.Popen(
							[
								ffmpeg,
								"-y",
								"-r",
								"30",
								"-fflags",
								"+genpts",
								"-i",
								str(h264),
								"-c",
								"copy",
								"-movflags",
								"+faststart",
								str(mp4),
							],
							stdout=subprocess.DEVNULL,
							stderr=subprocess.DEVNULL,
						)
		except Exception:
			# Muxing is optional; ignore any errors here.
			pass

		# Per-jump clip generation is now triggered at detection time (jobs are queued immediately)
		# and handled by an out-of-process worker. We keep session_stop lightweight.

		# Close IMU file
		try:
			if _imu_csv_fh:
				_imu_csv_fh.close()
		except Exception:
			pass
		_imu_csv_fh = None

		# Update session.json with stop time
		t_stop = time.time()
		try:
			if _session_dir:
				path = _session_dir / "session.json"
				if path.exists():
					data = json.loads(path.read_text(encoding="utf-8"))
				else:
					data = {}
				data["t_stop"] = t_stop
				path.write_text(json.dumps(data, indent=2), encoding="utf-8")
		except Exception:
			pass

		# Persist session stop + frames to DB (best-effort)
		try:
			await db.update_session_stop(
				session_id=sid,
				t_stop=t_stop,
				video_path=str(Path("data") / "sessions" / sid / "video.mp4"),
				meta=None,
			)
		except Exception:
			pass
		try:
			# Read frames.csv and bulk insert into DB
			if _session_dir:
				p = _session_dir / "frames.csv"
				if p.exists():
					out = []
					with open(p, "r", encoding="utf-8") as fh:
						_ = fh.readline()
						for line in fh:
							parts = line.strip().split(",")
							if len(parts) < 5:
								continue
							try:
								out.append(
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
					await db.replace_frames(sid, out)
		except Exception:
			pass

		_session_id = None
		_session_dir = None
		return {"detail": "Session stopped", "session_id": sid}


@app.get("/session/status")
async def session_status():
	return {"session_id": _session_id, "dir": str(_session_dir) if _session_dir else None}


@app.get("/debug/status")
async def debug_status():
	"""
	Quick diagnostics to tell whether IMU samples are flowing, detection is enabled,
	jump worker is alive, and whether the queue is dropping samples.
	"""
	q_size = None
	q_max = None
	try:
		if _jump_sample_queue is not None:
			q_size = _jump_sample_queue.qsize()
			q_max = getattr(_jump_sample_queue, "_maxsize", None)
	except Exception:
		pass
	# Clip worker queue depth (file-queue)
	try:
		_dbg["clip_jobs_pending"] = int(_count_clip_jobs_pending())
	except Exception:
		pass

	return {
		"detection_enabled": bool(_jump_detection_enabled),
		"active_mode": _active_mode,
		"active_rate": _active_rate,
		"jump_config": _jump_config,
		"session_id": _session_id,
		"queue_size": q_size,
		"queue_maxsize": q_max,
		"dbg": dict(_dbg),
	}


@app.get("/db/jumps")
async def db_list_jumps(limit: int = 200):
	"""
	List recent jumps from PostgreSQL (ordered by detection time DESC).
	If DB is not configured, returns an empty list.
	"""
	try:
		rows = await db.list_jumps(limit=limit)
		return {"count": len(rows), "jumps": rows}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB list_jumps failed: {e!r}")


@app.get("/db/jumps/{event_id}")
async def db_get_jump(event_id: int):
	"""
	Fetch one jump + IMU samples from PostgreSQL by event_id.
	"""
	try:
		row = await db.get_jump_with_imu(event_id)
		if not row:
			raise HTTPException(status_code=404, detail="Jump not found")
		return row
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB get_jump failed: {e!r}")


@app.delete("/db/jumps/{event_id}")
async def db_delete_jump(event_id: int):
	"""
	Delete the selected jump from DB, along with associated IMU samples.
	"""
	try:
		out = await db.delete_jump(int(event_id))
		return out
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB delete_jump failed: {e!r}")


def _session_base_dir(session_id: str) -> Path:
	# Session directories are stored under data/sessions/<session_id>/
	return Path("data") / "sessions" / session_id


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
		import shutil
		import subprocess

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

		ffmpeg = shutil.which("ffmpeg")
		if not ffmpeg:
			return

		clips_dir = base / "clips"
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

				subprocess.run(
					[
						ffmpeg,
						"-y",
						"-ss",
						f"{start_sec:.3f}",
						"-t",
						f"{duration:.3f}",
						"-i",
						str(mp4),
						"-c",
						"copy",
						"-movflags",
						"+faststart",
						str(out_path),
					],
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL,
					check=False,
				)

				if out_path.exists() and out_path.stat().st_size > 0:
					rel = str(Path("data") / "sessions" / session_id / "clips" / out_name)
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


@app.get("/sessions/{session_id}/video")
async def get_session_video(session_id: str):
	"""
	Serve a session video file for playback in the browser.
	Prefers MP4 (video.mp4). If missing, returns 404 and UI can instruct conversion.
	"""
	base = _session_base_dir(session_id)
	mp4 = base / "video.mp4"
	if mp4.exists():
		return FileResponse(str(mp4), media_type="video/mp4", filename="video.mp4")
	raise HTTPException(status_code=404, detail=f"video.mp4 not found for session {session_id!r}. Convert from video.h264 first.")


@app.get("/files")
async def get_file(path: str):
	"""
	Serve a file under the workspace (restricted to data/ only).
	Used for per-jump clip playback where DB stores a relative path like:
	  data/sessions/<sid>/clips/jump_<event_id>.mp4
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


@app.get("/sessions/{session_id}/frames")
async def get_session_frames(session_id: str, t0: float = None, t1: float = None, limit: int = 200000):
	"""
	Return per-frame timing as JSON for sync mapping.
	Prefers DB (frames table) if available, falls back to frames.csv on disk.
	"""
	# DB-first
	try:
		frames = await db.get_frames(session_id=session_id, limit=limit, t0=t0, t1=t1)
		if frames:
			return {"count": len(frames), "frames": frames, "source": "db"}
	except Exception:
		pass

	base = _session_base_dir(session_id)
	p = base / "frames.csv"
	if not p.exists():
		raise HTTPException(status_code=404, detail="frames.csv not found")
	# Keep it simple: parse CSV quickly without external deps.
	out = []
	try:
		with open(p, "r", encoding="utf-8") as fh:
			header = fh.readline()
			for line in fh:
				parts = line.strip().split(",")
				if len(parts) < 5:
					continue
				try:
					out.append(
						{
							"frame_idx": int(parts[0]),
							"t_host": float(parts[1]),
							"device_ts": float(parts[2]) if parts[2] else None,
							"width": int(parts[3]),
							"height": int(parts[4]),
						}
					)
				except Exception:
					continue
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to read frames.csv: {e!r}")
	return {"count": len(out), "frames": out, "source": "file"}


@app.post("/connect")
async def connect_device(payload: Dict[str, Any]):
	"""
	Start the IMU collector process connected to the specified device (MAC or name).
	The collector streams IMU packets to this server over localhost UDP.
	If a collector is already running, it will be stopped and replaced.
	"""
	global _imu_proc, _active_rate, _active_mode

	try:
		device = (payload.get("device") or "").strip()
		mode = (payload.get("mode") or MODE).strip().upper()
		rate = int(payload.get("rate") or RATE)
		# Persist the active stream parameters for the analysis worker.
		_active_rate = rate
		_active_mode = mode


		if not device:
			raise HTTPException(status_code=400, detail="Missing 'device' in request body.")

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
	try:
		if _detection_session_id is None:
			sid = time.strftime("%Y%m%d_%H%M%S") + "_detect"
			_detection_session_id = sid
			await session_start({"session_id": sid})
			_log_to_clients(f"[Session] Auto-started recording session for detection: {sid}")
	except Exception as e:
		_log_to_clients(f"[Session] Auto-start for detection failed: {e!r}")

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