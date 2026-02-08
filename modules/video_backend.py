from __future__ import annotations

import asyncio
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from modules.config import AppConfig, get_config


class VideoBackend(ABC):
	@abstractmethod
	def name(self) -> str: ...

	@abstractmethod
	def start(self) -> None: ...

	@abstractmethod
	def stop(self) -> None: ...

	@abstractmethod
	def get_status(self) -> Dict[str, Any]: ...

	@abstractmethod
	def start_recording(self, session_dir: str, fps: int) -> None: ...

	@abstractmethod
	def stop_recording(self) -> None: ...

	@abstractmethod
	def get_latest_jpeg(self) -> tuple[Optional[bytes], Optional[float]]: ...

	@abstractmethod
	def get_frames_since(self, since_t_host: float, include_frame_idx: bool = True, include_width_height: bool = True) -> list[Dict[str, Any]]:
		"""
		Retrieve frames with t_host >= since_t_host. Thread-safe.
		Returns list of frame dicts with t_host, device_ts, and optional frame_idx, width, height.
		"""
		...

	@abstractmethod
	async def mjpeg_stream(self, fps: float) -> AsyncIterator[bytes]: ...

	@abstractmethod
	async def snapshot_jpeg(self) -> Optional[bytes]: ...
	async def snapshot_jpeg(self) -> Optional[bytes]: ...

	@abstractmethod
	def mux_to_mp4_best_effort(self, session_dir: Path, fps: int) -> bool: ...

	@abstractmethod
	def cut_clip_best_effort(self, mp4_path: Path, out_path: Path, start_sec: float, duration_sec: float) -> bool: ...


def _truthy(v: str) -> bool:
	return v.strip().lower() in ("1", "true", "yes", "on")


def get_video_backend(cfg: Optional[AppConfig] = None, *, backend_override: Optional[str] = None) -> VideoBackend:
	# Default to Picamera2 single-pipeline backend (preview + record simultaneously).
	cfg = cfg or get_config()
	backend = (backend_override or cfg.video.backend or "picamera2").strip().lower()
	use_proc = bool(cfg.video.process.enabled)
	if use_proc:
		from modules.video_backends.http_proxy_backend import HttpProxyVideoBackend

		host = cfg.video.process.collector_host or "127.0.0.1"
		port = int(cfg.video.process.collector_port or 18081)
		return HttpProxyVideoBackend(f"http://{host}:{port}")

	if backend in ("picamera2", "pc2"):
		from modules.video_backends.picamera2_backend import Picamera2Backend

		# NOTE: do not use `or 1` here; camera index 0 is valid and would be overwritten.
		primary_idx = int(cfg.video.picamera2.primary_index)
		return Picamera2Backend(
			camera_index=primary_idx,
			label="module3",
			initial_config={
				"record_size": cfg.video.picamera2.module3.record_size,
				"preview_size": cfg.video.picamera2.module3.preview_size,
				"preview_fps": cfg.video.picamera2.module3.preview_fps,
				"bitrate": cfg.video.picamera2.module3.bitrate,
				"controls": cfg.video.picamera2.module3.controls,
			},
		)
	if backend == "jetson":
		from modules.video_backends.gstreamer_backend import JetsonGStreamerBackend

		return JetsonGStreamerBackend()

	# If an unknown backend is requested, fall back to picamera2 to avoid
	# silently using an unsupported legacy pipeline.
	from modules.video_backends.picamera2_backend import Picamera2Backend

	# NOTE: do not use `or 1` here; camera index 0 is valid and would be overwritten.
	primary_idx = int(cfg.video.picamera2.primary_index)
	return Picamera2Backend(
		camera_index=primary_idx,
		label="module3",
		initial_config={
			"record_size": cfg.video.picamera2.module3.record_size,
			"preview_size": cfg.video.picamera2.module3.preview_size,
			"preview_fps": cfg.video.picamera2.module3.preview_fps,
			"bitrate": cfg.video.picamera2.module3.bitrate,
			"controls": cfg.video.picamera2.module3.controls,
		},
	)


def get_video_backend_for_tools() -> VideoBackend:
	"""
	Return a backend instance intended for *file operations* (mux/cut) only.
	This intentionally ignores config.video.process to avoid proxying from worker processes.
	"""
	from modules.video_backends.ffmpeg_tools_backend import FfmpegToolsBackend

	return FfmpegToolsBackend()


def start_video_collector_subprocess(cfg: Optional[AppConfig] = None) -> Optional["subprocess.Popen"]:
	"""
	Optional helper: if config.video.process.enabled, start a separate process that owns the camera backend.
	"""
	cfg = cfg or get_config()
	if not bool(cfg.video.process.enabled):
		return None
	import subprocess

	port = int(cfg.video.process.collector_port or 18081)
	host = cfg.video.process.collector_host or "127.0.0.1"
	backend = (cfg.video.backend or "picamera2").strip().lower()
	cmd = [sys.executable, "-m", "modules.video_collector", "--host", str(host), "--port", str(port), "--backend", backend]
	try:
		return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except Exception:
		return None


async def mjpeg_from_latest(get_latest_jpeg_fn, fps: float) -> AsyncIterator[bytes]:
	"""
	Reusable MJPEG generator for backends that expose get_latest_jpeg().
	Yields full multipart chunks including boundary and headers.
	"""
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
		jpeg, t = get_latest_jpeg_fn()
		if jpeg is None or t is None:
			await asyncio.sleep(0.05)
			continue
		if last_t is not None and t == last_t:
			await asyncio.sleep(0.01)
			continue
		now_mono = time.monotonic()
		elapsed = now_mono - last_sent_mono
		if elapsed < min_interval:
			await asyncio.sleep(min_interval - elapsed)
			continue
		last_t = t
		last_sent_mono = time.monotonic()
		yield b"--" + boundary + b"\r\n"
		yield b"Content-Type: image/jpeg\r\n"
		yield b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
		yield jpeg + b"\r\n"


