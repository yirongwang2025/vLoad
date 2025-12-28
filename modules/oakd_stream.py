from __future__ import annotations

import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, TextIO


@dataclass
class OakdFrame:
	"""
	Holds the latest MJPEG frame and timing metadata.
	"""
	jpeg: bytes
	t_host: float
	device_ts: Optional[float] = None
	frame_idx: int = 0


class OakdStreamManager:
	"""
	DepthAI/OAK-D-S2 camera manager.

	MVP responsibilities:
	- Connect/disconnect device and run a DepthAI pipeline.
	- Provide a thread-safe "latest MJPEG frame" buffer for /video/mjpeg streaming.

	Notes:
	- Uses on-device VideoEncoder for MJPEG (stream) and H264 (future recording).
	- Requires `depthai` to be installed on the target (Raspberry Pi 5).
	"""

	def __init__(self) -> None:
		self._lock = threading.Lock()
		self._running = False
		self._thread: Optional[threading.Thread] = None
		self._latest: Optional[OakdFrame] = None
		self._last_error: Optional[str] = None

		# DepthAI objects live in the capture thread to keep teardown predictable.
		self._device = None

		# Recording state
		self._recording: bool = False
		self._record_dir: Optional[Path] = None
		self._video_fh: Optional[object] = None
		self._frames_fh: Optional[TextIO] = None
		self._record_fps: int = 30

		# Preview config (MJPEG stream)
		self._preview_size: Tuple[int, int] = (960, 540)  # good tradeoff for browser MJPEG
		self._preview_fps: int = 15

		# Record config (H.264 stream)
		self._record_size: Tuple[int, int] = (1920, 1080)
		self._record_stream_fps: int = 30

	def is_running(self) -> bool:
		with self._lock:
			return bool(self._running)

	def get_status(self) -> dict:
		with self._lock:
			return {
				"running": bool(self._running),
				"has_frame": self._latest is not None,
				"t_last_frame": self._latest.t_host if self._latest else None,
				"frame_idx": self._latest.frame_idx if self._latest else None,
				"recording": bool(self._recording),
				"record_dir": str(self._record_dir) if self._record_dir else None,
				"preview_size": list(self._preview_size),
				"preview_fps": int(self._preview_fps),
				"record_size": list(self._record_size),
				"record_fps": int(self._record_stream_fps),
				"error": self._last_error,
			}

	def get_latest_jpeg(self) -> Tuple[Optional[bytes], Optional[float]]:
		with self._lock:
			if self._latest is None:
				return None, None
			return self._latest.jpeg, self._latest.t_host

	def start(self) -> None:
		with self._lock:
			if self._running:
				return
			self._running = True
			self._last_error = None

		t = threading.Thread(target=self._run_capture_loop, name="oakd-capture", daemon=True)
		self._thread = t
		t.start()

	def start_recording(self, session_dir: str, fps: int = 30) -> None:
		"""
		Enable on-device H264 recording to files under session_dir.
		This does not start the camera; call start() first if needed.
		"""
		p = Path(session_dir)
		p.mkdir(parents=True, exist_ok=True)
		with self._lock:
			self._recording = True
			self._record_dir = p
			self._record_fps = int(fps) if int(fps) > 0 else 30

			# Open files in append mode (safe if reconnecting)
			self._video_fh = open(p / "video.h264", "ab")
			self._frames_fh = open(p / "frames.csv", "a", encoding="utf-8", newline="\n")
			# Write header if file is empty
			try:
				if (p / "frames.csv").stat().st_size == 0:
					self._frames_fh.write("frame_idx,t_host,device_ts,width,height\n")
					self._frames_fh.flush()
			except Exception:
				pass

	def stop_recording(self) -> None:
		with self._lock:
			self._recording = False
			self._record_dir = None
			try:
				if self._video_fh:
					self._video_fh.close()
			except Exception:
				pass
			try:
				if self._frames_fh:
					self._frames_fh.close()
			except Exception:
				pass
			self._video_fh = None
			self._frames_fh = None

	def stop(self) -> None:
		with self._lock:
			self._running = False

		# Device close happens in the capture thread; we just wait briefly for exit.
		t = self._thread
		if t and t.is_alive():
			t.join(timeout=3.0)
		self._thread = None

	def _run_capture_loop(self) -> None:
		try:
			import depthai as dai  # type: ignore
		except Exception as e:
			with self._lock:
				self._last_error = f"depthai import failed: {e!r}"
				self._running = False
			return

		# Use the v3 pattern: Camera.requestOutput(...).createOutputQueue() and VideoEncoder.build(...).
		is_depthai_v3 = bool(hasattr(dai, "Pipeline") and hasattr(dai.Pipeline, "start") and hasattr(dai, "node") and hasattr(dai.node, "HostNode"))

		device = None
		try:
			if is_depthai_v3:
				# ---- DepthAI v3 host pipeline ----
				pipeline = dai.Pipeline()

				# Build camera on the RGB socket (OAK-D-S2).
				# Use dai.node.Camera API from v3 examples.
				cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.RGB)
				# Request two outputs:
				# - low-res for MJPEG preview (browser-friendly + lower bandwidth/CPU)
				# - full-res for H264 recording
				with self._lock:
					preview_w, preview_h = self._preview_size
					preview_fps = int(self._preview_fps)
					record_w, record_h = self._record_size
					record_fps = int(self._record_stream_fps)

				preview_out = cam.requestOutput((preview_w, preview_h), type=dai.ImgFrame.Type.NV12)
				record_out = cam.requestOutput((record_w, record_h), type=dai.ImgFrame.Type.NV12)

				# Best-effort raw queue for record timestamps (so frames.csv matches the recorded stream)
				q_record_raw = None
				try:
					q_record_raw = record_out.createOutputQueue()
				except Exception:
					try:
						q_record_raw = record_out.createOutputQueue  # type: ignore[assignment]
					except Exception:
						q_record_raw = None

				# On-device MJPEG encoding for streaming.
				encoded_mjpeg = pipeline.create(dai.node.VideoEncoder).build(
					preview_out,
					frameRate=preview_fps,
					profile=dai.VideoEncoderProperties.Profile.MJPEG,
				)
				# Encoded output queue (JPEG bytes per packet).
				try:
					q_mjpeg = encoded_mjpeg.out.createOutputQueue()
				except Exception:
					# Some builds expose createOutputQueue on the encoded object directly
					q_mjpeg = encoded_mjpeg.createOutputQueue()  # type: ignore[attr-defined]

				# On-device H264 encoding for recording (always available even if unused).
				encoded_h264 = pipeline.create(dai.node.VideoEncoder).build(
					record_out,
					frameRate=record_fps,
					profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
				)
				try:
					q_h264 = encoded_h264.out.createOutputQueue()
				except Exception:
					q_h264 = encoded_h264.createOutputQueue()  # type: ignore[attr-defined]

				# Start the pipeline (v3 style).
				pipeline.start()
				self._device = pipeline  # store something non-None for status/debug
			else:
				# If we ever run on legacy DepthAI, we can reintroduce XLinkOut-based pipeline here.
				raise RuntimeError("This DepthAI build does not support the v3 host pipeline API required for streaming.")

			frame_idx = 0
			record_frame_idx = 0
			while True:
				with self._lock:
					if not self._running:
						break

				got_any = False

				# ---- Preview MJPEG (low-res) ----
				pkt = None
				try:
					pkt = q_mjpeg.tryGet()
				except Exception:
					pkt = None
				if pkt is not None:
					got_any = True
					jpeg = pkt.getData().tobytes()
					t_host = time.time()
					device_ts = None
					try:
						ts = pkt.getTimestamp()
						device_ts = ts.total_seconds()
					except Exception:
						device_ts = None

					frame_idx += 1
					with self._lock:
						self._latest = OakdFrame(
							jpeg=jpeg,
							t_host=t_host,
							device_ts=device_ts,
							frame_idx=frame_idx,
						)

				# ---- Recording timestamping (full-res raw frames, best-effort) ----
				with self._lock:
					rec = bool(self._recording)
					video_fh = self._video_fh
					frames_fh = self._frames_fh
					record_w, record_h = self._record_size

				if rec and frames_fh is not None and q_record_raw is not None:
					try:
						rpkt = q_record_raw.tryGet()
					except Exception:
						rpkt = None
					if rpkt is not None:
						got_any = True
						t_host = time.time()
						device_ts = None
						try:
							rts = rpkt.getTimestamp()
							device_ts = rts.total_seconds()
						except Exception:
							device_ts = None
						record_frame_idx += 1
						try:
							frames_fh.write(
								f"{record_frame_idx},{t_host},{device_ts if device_ts is not None else ''},{record_w},{record_h}\n"
							)
							frames_fh.flush()
						except Exception:
							pass

				# ---- Recording data (H264 bitstream) ----
				if rec and video_fh is not None:
					try:
						while True:
							try:
								hpkt = q_h264.tryGet()
							except Exception:
								hpkt = None
							if hpkt is None:
								break
							got_any = True
							data = hpkt.getData().tobytes()
							if data:
								video_fh.write(data)
					except Exception:
						pass

				if not got_any:
					time.sleep(0.004)

		except Exception as e:
			with self._lock:
				self._last_error = f"camera loop error: {e!r}"
		finally:
			# Ensure recording files are closed if the loop exits
			try:
				self.stop_recording()
			except Exception:
				pass
			with self._lock:
				self._running = False
			try:
				# v3 pipeline stop
				if device is not None and hasattr(device, "close"):
					device.close()
				if device is not None and hasattr(device, "stop"):
					device.stop()
				if device is not None and hasattr(device, "wait"):
					device.wait()
			except Exception:
				pass
			self._device = None


