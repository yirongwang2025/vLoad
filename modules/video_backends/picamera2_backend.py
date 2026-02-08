from __future__ import annotations

import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from modules.video_backend import VideoBackend, mjpeg_from_latest


class Picamera2Backend(VideoBackend):
	"""
	Picamera2/libcamera backend that supports *simultaneous*:
	- low-res preview frames (JPEG) for browser MJPEG
	- high-res H264 recording to disk

	This avoids the "two rpicam-vid processes fight over one CSI camera" problem.

	Notes:
	- `python3-picamera2` is a system package on Raspberry Pi OS (apt).
	- JPEG encoding uses Pillow (`PIL`), installed via pip (requirements.txt).
	"""

	def __init__(self, camera_index: Optional[int] = None, label: str = "picamera2", initial_config: Optional[Dict[str, Any]] = None) -> None:
		self._lock = threading.Lock()
		self._label = str(label or "picamera2")
		self._camera_index: Optional[int] = int(camera_index) if camera_index is not None else None

		self._running = False
		self._recording = False
		self._last_error: Optional[str] = None

		# latest preview frame
		self._latest_jpeg: Optional[bytes] = None
		self._latest_t_host: Optional[float] = None
		self._last_preview_encode_t: float = 0.0

		# recording state
		self._record_dir: Optional[Path] = None
		self._record_fps: int = 30
		self._record_size = (1920, 1080)
		self._preview_size = (960, 540)
		self._preview_fps: int = 15
		self._bitrate: int = 10_000_000
		self._extra_controls: dict[str, Any] = {}

		# Apply best-effort initial config from config.json (if provided).
		self._apply_initial_config(initial_config or {})
		self._frames: list[dict[str, Any]] = []

		# Picamera2 objects (lazy-imported)
		self._picam2 = None
		self._encoder = None
		self._output = None
		self._thread: Optional[threading.Thread] = None

	def name(self) -> str:
		return self._label

	def _apply_initial_config(self, cfg: Dict[str, Any]) -> None:
		"""
		Best-effort initial config. Invalid values are ignored (do not raise).
		Expected keys: record_size, preview_size, preview_fps, bitrate, controls
		"""

		def _parse_size_any(v: Any) -> Optional[tuple[int, int]]:
			try:
				if isinstance(v, (list, tuple)) and len(v) == 2:
					w = int(v[0])
					h = int(v[1])
					return (w, h) if w > 0 and h > 0 else None
				if isinstance(v, str):
					ss = v.lower().replace(" ", "")
					if "x" not in ss:
						return None
					a, b = ss.split("x", 1)
					w = int(a)
					h = int(b)
					return (w, h) if w > 0 and h > 0 else None
			except Exception:
				return None
			return None

		try:
			if not isinstance(cfg, dict):
				return
			rs = cfg.get("record_size")
			ps = cfg.get("preview_size")
			pfps = cfg.get("preview_fps")
			br = cfg.get("bitrate")
			controls = cfg.get("controls")

			if rs is not None:
				parsed = _parse_size_any(rs)
				if parsed:
					self._record_size = parsed
			if ps is not None:
				parsed = _parse_size_any(ps)
				if parsed:
					self._preview_size = parsed
			if pfps is not None:
				v = int(float(pfps))
				if v > 0:
					self._preview_fps = v
			if br is not None:
				v = int(float(br))
				if v > 0:
					self._bitrate = v
			if isinstance(controls, dict):
				self._extra_controls = dict(controls)
		except Exception:
			return

	def get_status(self) -> Dict[str, Any]:
		with self._lock:
			return {
				"label": self._label,
				"camera_index": self._camera_index,
				"running": bool(self._running),
				"preview_running": bool(self._running),
				"has_frame": self._latest_jpeg is not None,
				"t_last_frame": self._latest_t_host,
				"recording": bool(self._recording),
				"record_dir": str(self._record_dir) if self._record_dir else None,
				"preview_size": [int(self._preview_size[0]), int(self._preview_size[1])],
				"preview_fps": int(self._preview_fps),
				"record_size": [int(self._record_size[0]), int(self._record_size[1])],
				"record_fps": int(self._record_fps),
				"bitrate": int(self._bitrate),
				"controls": dict(self._extra_controls or {}),
				"error": self._last_error,
			}

	def get_config(self) -> Dict[str, Any]:
		"""
		Configuration that can be controlled at runtime (used by frontend config UI).
		"""
		with self._lock:
			return {
				"label": self._label,
				"camera_index": self._camera_index,
				"record_size": [int(self._record_size[0]), int(self._record_size[1])],
				"preview_size": [int(self._preview_size[0]), int(self._preview_size[1])],
				"preview_fps": int(self._preview_fps),
				"bitrate": int(self._bitrate),
				"controls": dict(self._extra_controls or {}),
			}

	def update_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Update config values. Some settings require a restart (resize/bitrate).
		Returns: {ok, restart_required, applied, warning?}
		"""

		def _parse_size_any(v: Any) -> Optional[tuple[int, int]]:
			try:
				if isinstance(v, (list, tuple)) and len(v) == 2:
					w = int(v[0])
					h = int(v[1])
					return (w, h) if w > 0 and h > 0 else None
				if isinstance(v, str):
					ss = v.lower().replace(" ", "")
					if "x" not in ss:
						return None
					a, b = ss.split("x", 1)
					w = int(a)
					h = int(b)
					return (w, h) if w > 0 and h > 0 else None
			except Exception:
				return None
			return None

		if not isinstance(cfg, dict):
			return {"ok": False, "error": "config payload must be a JSON object"}

		restart_required = False
		warnings: list[str] = []

		with self._lock:
			# Keep originals for restart decision
			orig_record_size = self._record_size
			orig_preview_size = self._preview_size
			orig_bitrate = self._bitrate

			if "record_size" in cfg:
				parsed = _parse_size_any(cfg.get("record_size"))
				if not parsed:
					return {"ok": False, "error": "record_size must be [w,h] or \"WxH\" (e.g. \"1456x1088\")"}
				self._record_size = parsed
			if "preview_size" in cfg:
				parsed = _parse_size_any(cfg.get("preview_size"))
				if not parsed:
					return {"ok": False, "error": "preview_size must be [w,h] or \"WxH\" (e.g. \"1280x720\")"}
				self._preview_size = parsed
			if "preview_fps" in cfg:
				try:
					v = int(float(cfg.get("preview_fps")))
					if v <= 0:
						return {"ok": False, "error": "preview_fps must be > 0"}
					self._preview_fps = v
				except Exception:
					return {"ok": False, "error": "preview_fps must be a number"}
			if "bitrate" in cfg:
				try:
					v = int(float(cfg.get("bitrate")))
					if v <= 0:
						return {"ok": False, "error": "bitrate must be > 0"}
					self._bitrate = v
				except Exception:
					return {"ok": False, "error": "bitrate must be a number"}
			if "controls" in cfg:
				controls = cfg.get("controls")
				if controls is None:
					self._extra_controls = {}
				elif not isinstance(controls, dict):
					return {"ok": False, "error": "controls must be a JSON object (dictionary) or null"}
				else:
					# Best-effort: store values; actual application depends on libcamera support.
					self._extra_controls = dict(controls)

			# Decide whether restart is required.
			if self._record_size != orig_record_size or self._preview_size != orig_preview_size or self._bitrate != orig_bitrate:
				restart_required = True

			picam2 = self._picam2
			running = bool(self._running)
			recording = bool(self._recording)
			controls_to_apply = dict(self._extra_controls or {})

		# Apply controls live if possible (best-effort). Resizes/bitrate require restart.
		if running and not restart_required and controls_to_apply and picam2 is not None and not recording:
			try:
				picam2.set_controls(controls_to_apply)
			except Exception as e:
				warnings.append(f"Failed to apply controls live (will apply on restart): {e!r}")

		out: Dict[str, Any] = {
			"ok": True,
			"restart_required": bool(restart_required),
			"applied": self.get_config(),
		}
		if warnings:
			out["warning"] = "; ".join(warnings)
		return out

	def start(self) -> None:
		with self._lock:
			if self._running:
				return
			self._running = True
			self._last_error = None

		t = threading.Thread(target=self._run_loop, name=f"{self._label}-picamera2", daemon=True)
		self._thread = t
		t.start()

	def stop(self) -> None:
		with self._lock:
			self._running = False
		try:
			self.stop_recording()
		except Exception:
			pass
		t = self._thread
		if t and t.is_alive():
			t.join(timeout=2.0)
		self._thread = None

	def _run_loop(self) -> None:
		try:
			from picamera2 import Picamera2  # type: ignore
			from picamera2.encoders import H264Encoder  # type: ignore
			from picamera2.outputs import FileOutput  # type: ignore
		except Exception as e:
			import sys
			with self._lock:
				self._last_error = (
					f"Picamera2 import failed: {e!r}. "
					f"Python={sys.executable!r}. "
					"Common cause: server is running inside a venv that does NOT include system site-packages, "
					"so the apt-installed `python3-picamera2` is not visible. "
					"Fix: recreate venv with `python3 -m venv --system-site-packages <venv>` (or run with system python)."
				)
				self._running = False
			return

		try:
			from PIL import Image  # type: ignore
		except Exception as e:
			with self._lock:
				self._last_error = f"Pillow import failed: {e!r}. Install `Pillow` (pip)."
				self._running = False
			return

		# Build camera instance
		try:
			# Helpful diagnostics: show camera count if index selection fails.
			try:
				infos = Picamera2.global_camera_info()  # type: ignore[attr-defined]
			except Exception:
				infos = None

			if self._camera_index is None:
				picam2 = Picamera2()
			else:
				# Picamera2 uses camera_num to select camera
				if isinstance(infos, list) and len(infos) == 0:
					raise IndexError("no cameras detected (global_camera_info empty)")
				if isinstance(infos, list) and int(self._camera_index) >= len(infos):
					raise IndexError(f"camera_index={int(self._camera_index)} out of range (found {len(infos)} camera(s))")
				picam2 = Picamera2(camera_num=int(self._camera_index))
		except Exception as e:
			with self._lock:
				self._last_error = f"Picamera2 init failed: {e!r}"
				self._running = False
			return

		# Configure dual streams: main (record) + lores (preview)
		try:
			record_w, record_h = self._record_size
			prev_w, prev_h = self._preview_size
			controls: dict[str, Any] = {"FrameRate": float(self._record_fps)}
			# Merge extra controls (useful for GS modules: NR/sharpness/exposure/gain, etc).
			# If a control is not supported by the camera, Picamera2 may raise on configure/set_controls,
			# so we keep it best-effort (and fall back to defaults on error below).
			for k, v in (self._extra_controls or {}).items():
				if k and v is not None:
					controls[str(k)] = v
			cfg = picam2.create_video_configuration(
				main={"size": (int(record_w), int(record_h)), "format": "YUV420"},
				lores={"size": (int(prev_w), int(prev_h)), "format": "RGB888"},
				controls=controls,
			)
			picam2.configure(cfg)
			# Some Picamera2 versions apply controls more reliably after configure.
			try:
				if controls:
					picam2.set_controls(controls)
			except Exception:
				pass
		except Exception as e:
			with self._lock:
				self._last_error = f"Picamera2 configure failed: {e!r}"
				self._running = False
			try:
				picam2.close()
			except Exception:
				pass
			return

		encoder = H264Encoder(bitrate=int(self._bitrate))

		# callback called for each completed request
		def _on_request(request) -> None:  # type: ignore[no-untyped-def]
			now = time.time()
			# Preview throttling
			try:
				with self._lock:
					target_dt = 1.0 / max(1.0, float(self._preview_fps))
					if (now - self._last_preview_encode_t) < target_dt:
						do_preview = False
					else:
						self._last_preview_encode_t = now
						do_preview = True
			except Exception:
				do_preview = True

			if do_preview:
				try:
					arr = request.make_array("lores")
					im = Image.fromarray(arr)
					buf = BytesIO()
					im.save(buf, format="JPEG", quality=80, optimize=True)
					jpg = buf.getvalue()
					with self._lock:
						self._latest_jpeg = jpg
						self._latest_t_host = now
				except Exception:
					# Keep preview best-effort; don't kill the pipeline on encode errors.
					pass

			# Recording frame timing (host time + optional sensor timestamp)
			try:
				with self._lock:
					rec = bool(self._recording)
				if not rec:
					return
				meta = request.get_metadata()
				sensor_ns = meta.get("SensorTimestamp") if isinstance(meta, dict) else None
				self._frames.append(
					{
						"t_host": float(now),
						"device_ts": (float(sensor_ns) / 1e9) if isinstance(sensor_ns, (int, float)) else None,
					}
				)
			except Exception:
				pass

		try:
			picam2.post_callback = _on_request
		except Exception:
			# Fallback: older Picamera2 versions might not support post_callback;
			# pipeline still works, but we won't have preview/frames.
			pass

		# Start camera running
		try:
			picam2.start()
		except Exception as e:
			with self._lock:
				self._last_error = f"Picamera2 start failed: {e!r}"
				self._running = False
			try:
				picam2.close()
			except Exception:
				pass
			return

		with self._lock:
			self._picam2 = picam2
			self._encoder = encoder
			self._output = FileOutput  # store class for later (avoid reimport)

		try:
			while True:
				with self._lock:
					if not self._running:
						break
				time.sleep(0.05)
		finally:
			try:
				try:
					picam2.stop_recording()
				except Exception:
					pass
				picam2.stop()
			except Exception:
				pass
			try:
				picam2.close()
			except Exception:
				pass
			with self._lock:
				self._picam2 = None
				self._encoder = None
				self._output = None

	def start_recording(self, session_dir: str, fps: int) -> None:
		base = Path(session_dir)
		base.mkdir(parents=True, exist_ok=True)
		video_h264 = base / "video.h264"

		with self._lock:
			self._record_fps = int(fps) if int(fps) > 0 else 30
			self._recording = True
			self._record_dir = base
			self._frames = []
			picam2 = self._picam2
			encoder = self._encoder
			FileOutput = self._output

		if picam2 is None or encoder is None or FileOutput is None:
			with self._lock:
				self._recording = False
				self._record_dir = None
				self._last_error = "Camera not running. Call /video/connect first (or ensure backend auto-starts)."
			return

		try:
			# Start recording without stopping preview
			picam2.start_recording(encoder, FileOutput(str(video_h264)))
		except Exception as e:
			with self._lock:
				self._recording = False
				self._record_dir = None
				self._last_error = f"start_recording failed: {e!r}"
			return

	def stop_recording(self) -> None:
		with self._lock:
			picam2 = self._picam2
			base = self._record_dir
			w, h = self._record_size
			frames = list(self._frames)
			self._frames = []
			self._recording = False
			self._record_dir = None

		if picam2 is None:
			return

		try:
			picam2.stop_recording()
		except Exception:
			pass

		# Write frames.csv (best-effort) for sync
		if base is None:
			return
		try:
			frames_path = base / "frames.csv"
			out = ["frame_idx,t_host,device_ts,width,height"]
			for i, f in enumerate(frames):
				try:
					th = float(f.get("t_host"))
				except Exception:
					continue
				dts = f.get("device_ts")
				out.append(
					f"{int(i)},{th},{dts if dts is not None else ''},{int(w)},{int(h)}"
				)
			frames_path.write_text("\n".join(out) + "\n", encoding="utf-8")
		except Exception:
			pass

	def get_latest_jpeg(self) -> tuple[Optional[bytes], Optional[float]]:
		with self._lock:
			return self._latest_jpeg, self._latest_t_host

	def get_frames_since(self, since_t_host: float, include_frame_idx: bool = True, include_width_height: bool = True) -> list[dict[str, Any]]:
		"""
		Thread-safe method to retrieve frames with t_host >= since_t_host.
		Returns a list of frame dictionaries with t_host and optional device_ts, frame_idx, width, height.
		"""
		with self._lock:
			if not isinstance(self._frames, list):
				return []
			w, h = self._record_size
			out = []
			for i, f in enumerate(self._frames):
				try:
					t_host = float(f.get("t_host", 0.0))
					if t_host < since_t_host:
						continue
					frame_data: dict[str, Any] = {
						"t_host": t_host,
						"device_ts": f.get("device_ts"),
					}
					if include_frame_idx:
						frame_data["frame_idx"] = int(i)
					if include_width_height:
						frame_data["width"] = int(w)
						frame_data["height"] = int(h)
					out.append(frame_data)
				except Exception:
					continue
			return out

	async def mjpeg_stream(self, fps: float) -> AsyncIterator[bytes]:
		async for chunk in mjpeg_from_latest(self.get_latest_jpeg, fps=float(fps)):
			yield chunk

	async def snapshot_jpeg(self) -> Optional[bytes]:
		jpeg, _t = self.get_latest_jpeg()
		return jpeg

	def mux_to_mp4_best_effort(self, session_dir: Path, fps: int) -> bool:
		# Reuse existing ffmpeg tools via the legacy helpers
		from modules.video_tools import mux_h264_to_mp4_async

		h264 = session_dir / "video.h264"
		mp4 = session_dir / "video.mp4"
		return mux_h264_to_mp4_async(h264, mp4, fps=int(fps))

	def cut_clip_best_effort(self, mp4_path: Path, out_path: Path, start_sec: float, duration_sec: float) -> bool:
		from modules.video_tools import cut_mp4_clip_copy

		return cut_mp4_clip_copy(mp4_path, out_path, start_sec=float(start_sec), duration_sec=float(duration_sec))


