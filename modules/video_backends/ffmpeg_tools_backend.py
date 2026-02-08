from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, List

from modules.video_backend import VideoBackend
from modules.video_tools import cut_mp4_clip_copy, mux_h264_to_mp4_async


class FfmpegToolsBackend(VideoBackend):
	"""
	Backend used only for file-based video operations (mux/cut) so worker processes
	don't depend on a specific camera backend (OAK-D vs PiCam vs Jetson).
	"""

	def name(self) -> str:
		return "ffmpeg-tools"

	def start(self) -> None:
		return

	def stop(self) -> None:
		return

	def get_status(self) -> Dict[str, Any]:
		return {"running": False}

	def start_recording(self, session_dir: str, fps: int) -> None:
		raise RuntimeError("FfmpegToolsBackend does not support capture/recording")

	def stop_recording(self) -> None:
		return

	def get_latest_jpeg(self) -> tuple[Optional[bytes], Optional[float]]:
		return None, None

	def get_frames_since(self, since_t_host: float, include_frame_idx: bool = True, include_width_height: bool = True) -> List[Dict[str, Any]]:
		# FfmpegToolsBackend is for file operations only, not capture
		return []

	async def mjpeg_stream(self, fps: float) -> AsyncIterator[bytes]:
		if False:  # pragma: no cover
			yield b""
		return

	async def snapshot_jpeg(self) -> Optional[bytes]:
		return None

	def mux_to_mp4_best_effort(self, session_dir: Path, fps: int) -> bool:
		h264 = session_dir / "video.h264"
		mp4 = session_dir / "video.mp4"
		return mux_h264_to_mp4_async(h264, mp4, fps=int(fps))

	def cut_clip_best_effort(self, mp4_path: Path, out_path: Path, start_sec: float, duration_sec: float) -> bool:
		return cut_mp4_clip_copy(mp4_path, out_path, start_sec=float(start_sec), duration_sec=float(duration_sec))


