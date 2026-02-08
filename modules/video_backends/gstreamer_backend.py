from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from modules.video_backend import VideoBackend


class JetsonGStreamerBackend(VideoBackend):
	"""
	Stub backend to make Jetson migration a clean “add implementation” task.
	When you migrate, implement this backend using GStreamer (nvarguscamerasrc / v4l2src,
	nvv4l2h264enc, etc.) and keep server/UI unchanged.
	"""

	def name(self) -> str:
		return "jetson"

	def start(self) -> None:
		raise NotImplementedError("Jetson backend not implemented yet")

	def stop(self) -> None:
		return

	def get_status(self) -> Dict[str, Any]:
		return {"running": False, "error": "jetson backend not implemented"}

	def start_recording(self, session_dir: str, fps: int) -> None:
		raise NotImplementedError("Jetson backend not implemented yet")

	def stop_recording(self) -> None:
		return

	def get_latest_jpeg(self) -> tuple[Optional[bytes], Optional[float]]:
		return None, None

	def get_frames_since(self, since_t_host: float, include_frame_idx: bool = True, include_width_height: bool = True) -> list[Dict[str, Any]]:
		# Stub: Jetson backend not implemented yet
		return []

	async def mjpeg_stream(self, fps: float) -> AsyncIterator[bytes]:
		# no-op stream
		if False:  # pragma: no cover
			yield b""
		return

	async def snapshot_jpeg(self) -> Optional[bytes]:
		return None

	def mux_to_mp4_best_effort(self, session_dir: Path, fps: int) -> bool:
		# Jetson implementation may use gst instead of ffmpeg
		return False

	def cut_clip_best_effort(self, mp4_path: Path, out_path: Path, start_sec: float, duration_sec: float) -> bool:
		return False


