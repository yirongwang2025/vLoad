from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


def find_ffmpeg() -> Optional[str]:
	return shutil.which("ffmpeg")


def find_ffprobe() -> Optional[str]:
	return shutil.which("ffprobe")


def probe_mp4_stream_info(mp4_path: Path) -> dict[str, Any]:
	"""
	Return basic video stream info via ffprobe (width/height/fps if available).
	Best-effort; returns {} on failure.
	"""
	ffprobe = find_ffprobe()
	if not ffprobe or not mp4_path.exists():
		return {}
	try:
		p = subprocess.run(
			[
				ffprobe,
				"-v",
				"error",
				"-select_streams",
				"v:0",
				"-show_entries",
				"stream=width,height,r_frame_rate,avg_frame_rate",
				"-of",
				"json",
				str(mp4_path),
			],
			stdout=subprocess.PIPE,
			stderr=subprocess.DEVNULL,
			check=False,
			text=True,
		)
		data = json.loads(p.stdout or "{}")
		streams = data.get("streams") or []
		if not streams:
			return {}
		s0 = streams[0] if isinstance(streams[0], dict) else {}
		return {
			"width": s0.get("width"),
			"height": s0.get("height"),
			"r_frame_rate": s0.get("r_frame_rate"),
			"avg_frame_rate": s0.get("avg_frame_rate"),
		}
	except Exception:
		return {}


def extract_mp4_frame_times(mp4_path: Path) -> list[float]:
	"""
	Extract per-frame timestamps (seconds, clip-relative) from an MP4 using ffprobe.
	Intended for SHORT clips (jump clips), not full-session videos.
	"""
	ffprobe = find_ffprobe()
	if not ffprobe or not mp4_path.exists():
		return []
	try:
		p = subprocess.run(
			[
				ffprobe,
				"-v",
				"error",
				"-select_streams",
				"v:0",
				"-show_frames",
				"-show_entries",
				"frame=best_effort_timestamp_time,pkt_pts_time,pkt_dts_time",
				"-of",
				"json",
				str(mp4_path),
			],
			stdout=subprocess.PIPE,
			stderr=subprocess.DEVNULL,
			check=False,
			text=True,
		)
		data = json.loads(p.stdout or "{}")
		frames = data.get("frames") or []
		out: list[float] = []
		for fr in frames:
			if not isinstance(fr, dict):
				continue
			# Prefer best-effort timestamps
			for k in ("best_effort_timestamp_time", "pkt_pts_time", "pkt_dts_time"):
				v = fr.get(k)
				if v is None or v == "N/A":
					continue
				try:
					out.append(float(v))
					break
				except Exception:
					continue
		# Ensure monotonic non-decreasing (ffprobe can produce tiny negatives)
		out.sort()
		return out
	except Exception:
		return []


def mux_h264_to_mp4_async(h264_path: Path, mp4_path: Path, fps: int) -> bool:
	"""
	Best-effort mux: raw H264 elementary stream -> MP4 container.
	Non-blocking (starts ffmpeg subprocess and returns).
	"""
	ffmpeg = find_ffmpeg()
	if not ffmpeg:
		return False
	if not h264_path.exists():
		return False
	if mp4_path.exists():
		return True

	subprocess.Popen(
		[
			ffmpeg,
			"-y",
			"-r",
			str(int(fps)),
			"-fflags",
			"+genpts",
			"-i",
			str(h264_path),
			"-c",
			"copy",
			"-movflags",
			"+faststart",
			str(mp4_path),
		],
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)
	return True


def cut_mp4_clip_copy(mp4_path: Path, out_path: Path, start_sec: float, duration_sec: float) -> bool:
	"""
	Best-effort clip cut using stream copy (fast). Returns True if output exists and is non-empty.
	"""
	ffmpeg = find_ffmpeg()
	if not ffmpeg:
		return False
	if not mp4_path.exists():
		return False
	out_path.parent.mkdir(parents=True, exist_ok=True)

	subprocess.run(
		[
			ffmpeg,
			"-y",
			"-ss",
			f"{float(start_sec):.3f}",
			"-t",
			f"{float(duration_sec):.3f}",
			"-i",
			str(mp4_path),
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

	return bool(out_path.exists() and out_path.stat().st_size > 0)


def cut_h264_clip_to_mp4_best_effort(
	h264_path: Path,
	out_path: Path,
	start_sec: float,
	duration_sec: float,
	fps: int,
) -> bool:
	"""
	Best-effort "real-time" clip from a raw H264 elementary stream.

	This is used when the session is still running and video.mp4 is not available yet.
	Because H264 elementary streams don't always cut cleanly with stream-copy, we:
	- try a fast stream-copy into MP4 first
	- fall back to re-encode (more compatible, more CPU)
	"""
	ffmpeg = find_ffmpeg()
	if not ffmpeg:
		return False
	if not h264_path.exists():
		return False
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fps_i = int(fps) if int(fps) > 0 else 30

	# Attempt 1: stream-copy (fast, may fail depending on GOP / timing / missing timestamps)
	subprocess.run(
		[
			ffmpeg,
			"-y",
			"-r",
			str(fps_i),
			"-fflags",
			"+genpts",
			"-ss",
			f"{float(start_sec):.3f}",
			"-t",
			f"{float(duration_sec):.3f}",
			"-i",
			str(h264_path),
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
		return True

	# Attempt 2: re-encode (slower, but more robust)
	try:
		if out_path.exists():
			out_path.unlink()
	except Exception:
		pass
	subprocess.run(
		[
			ffmpeg,
			"-y",
			"-r",
			str(fps_i),
			"-fflags",
			"+genpts",
			"-ss",
			f"{float(start_sec):.3f}",
			"-t",
			f"{float(duration_sec):.3f}",
			"-i",
			str(h264_path),
			"-an",
			"-c:v",
			"libx264",
			"-preset",
			"ultrafast",
			"-crf",
			"23",
			"-pix_fmt",
			"yuv420p",
			"-movflags",
			"+faststart",
			str(out_path),
		],
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
		check=False,
	)
	return bool(out_path.exists() and out_path.stat().st_size > 0)


