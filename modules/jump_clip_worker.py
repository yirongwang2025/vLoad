from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any, Dict, Optional

from modules import db
from modules.config import get_config
from modules.video_backend import get_video_backend_for_tools
from modules.video_tools import cut_h264_clip_to_mp4_best_effort, extract_mp4_frame_times, probe_mp4_stream_info


def _safe_read_json(path: Path) -> Dict[str, Any]:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return {}


def _write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding="utf-8")


def _parse_enqueued_ms_from_name(path: Path) -> Optional[int]:
	"""
	Best-effort parser for job filenames:
	  jump_<jump_id>_<enqueued_ms>.json
	  jump_<jump_id>_<enqueued_ms>.processing
	"""
	try:
		parts = path.stem.split("_")
		if len(parts) >= 3 and parts[-1].isdigit():
			return int(parts[-1])
	except Exception:
		return None
	return None


def _job_age_seconds(path: Path) -> float:
	now = time.time()
	ms = _parse_enqueued_ms_from_name(path)
	if ms is not None:
		return max(0.0, now - (float(ms) / 1000.0))
	try:
		return max(0.0, now - float(path.stat().st_mtime))
	except Exception:
		return 0.0


def _unique_failed_path(fail_dir: Path, original_name: str) -> Path:
	base = fail_dir / original_name
	if not base.exists():
		return base
	stamp = int(time.time() * 1000)
	return fail_dir / f"{Path(original_name).stem}_{stamp}{Path(original_name).suffix}"


def _move_to_failed(path: Path, fail_dir: Path, reason: str) -> None:
	try:
		target_name = path.name.replace(".processing", ".failed.json")
		target = _unique_failed_path(fail_dir, target_name)
		path.rename(target)
		_write_text(fail_dir / f"{target.stem}.error.txt", reason)
	except Exception:
		pass


async def _prune_startup_jobs(jobs_dir: Path, fail_dir: Path) -> None:
	"""
	One-time startup cleanup:
	- Drop invalid job payloads.
	- Drop jobs for missing session directories.
	- Drop stale jobs that exceeded wait timeout and still have no video source.
	- Recover orphan .processing files from prior crashes back to .json.
	"""
	cfg = get_config()
	sessions_base = Path(cfg.sessions.base_dir)
	default_wait_s = float(cfg.runtime.clip_wait_mp4_timeout_seconds)
	pruned_invalid = 0
	pruned_missing_session = 0
	pruned_stale = 0
	pruned_missing_db_row = 0
	recovered_processing = 0
	pruned_orphan_clips = 0

	candidates = list(jobs_dir.glob("*.json")) + list(jobs_dir.glob("*.processing"))
	for path in candidates:
		job = _safe_read_json(path)
		jump_id = int(job.get("jump_id") or 0)
		session_id = str(job.get("session_id") or "").strip()
		if jump_id <= 0 or not session_id:
			_move_to_failed(path, fail_dir, "startup-prune: invalid job payload")
			pruned_invalid += 1
			continue

		session_dir = sessions_base / session_id
		if not session_dir.exists():
			_move_to_failed(path, fail_dir, f"startup-prune: missing session directory: {session_dir}")
			pruned_missing_session += 1
			continue

		# If this exact jump_id no longer exists, drop queue items and any derived clip file.
		try:
			resolved = await db.resolve_jump_row_id(jump_id=int(jump_id))
		except Exception:
			resolved = None
		if not resolved:
			try:
				event_id = int(job.get("event_id") or 0)
			except Exception:
				event_id = 0
			jump_clips_subdir = cfg.sessions.jump_clips_subdir.strip()
			clips_dir = session_dir / jump_clips_subdir
			candidates = []
			if event_id > 0:
				candidates.append(clips_dir / f"jump_{event_id}.mp4")
			candidates.append(clips_dir / f"jump_{int(jump_id)}.mp4")
			for cp in candidates:
				try:
					if cp.exists():
						cp.unlink()
				except Exception:
					pass
			_move_to_failed(path, fail_dir, "startup-prune: jump row missing in DB")
			pruned_missing_db_row += 1
			continue

		wait_timeout_s = float(job.get("wait_mp4_timeout_s") or default_wait_s)
		age_s = _job_age_seconds(path)
		mp4 = session_dir / "video.mp4"
		h264 = session_dir / "video.h264"
		video_ready = bool((mp4.exists() and mp4.stat().st_size > 0) or (h264.exists() and h264.stat().st_size > 0))
		# Conservative stale threshold: at least 2x wait timeout + 60s.
		stale_threshold_s = max(120.0, (wait_timeout_s * 2.0) + 60.0)
		if (not video_ready) and age_s > stale_threshold_s:
			_move_to_failed(
				path,
				fail_dir,
				f"startup-prune: stale job (age={age_s:.1f}s > {stale_threshold_s:.1f}s) and no mp4/h264",
			)
			pruned_stale += 1
			continue

		# Recover orphan processing files left by abrupt shutdown.
		if path.suffix == ".processing":
			requeued = path.with_suffix(".json")
			try:
				if requeued.exists():
					# Keep the freshest pending; mark orphan as failed to avoid duplicates.
					_move_to_failed(path, fail_dir, "startup-prune: orphan .processing with existing .json")
				else:
					path.rename(requeued)
					recovered_processing += 1
			except Exception:
				pass

	# Remove orphan clip files that are no longer referenced by any jump row.
	try:
		for session_dir in sessions_base.glob("*"):
			if not session_dir.is_dir():
				continue
			clips_dir = session_dir / cfg.sessions.jump_clips_subdir.strip()
			if not clips_dir.exists():
				continue
			for clip_path in clips_dir.glob("jump_*.mp4"):
				rel = str((sessions_base / session_dir.name / cfg.sessions.jump_clips_subdir.strip() / clip_path.name))
				try:
					referenced = await db.is_video_path_referenced(rel)
				except Exception:
					referenced = True
				if not referenced:
					try:
						clip_path.unlink()
						pruned_orphan_clips += 1
					except Exception:
						pass
	except Exception:
		pass


def _read_session_t_start(session_dir: Path) -> Optional[float]:
	"""
	Read session start time (epoch seconds) from session.json.
	This is used to convert host-time windows into video-time offsets without needing frames.csv.
	"""
	try:
		p = session_dir / "session.json"
		if not p.exists():
			return None
		data = json.loads(p.read_text(encoding="utf-8"))
		t0 = data.get("t_start")
		return float(t0) if isinstance(t0, (int, float, str)) else None
	except Exception:
		return None


async def _wait_for_file(path: Path, timeout_s: float) -> bool:
	CFG = get_config()
	deadline = time.time() + float(timeout_s)
	while time.time() < deadline:
		if path.exists() and path.stat().st_size > 0:
			return True
		await asyncio.sleep(float(CFG.clip_worker.poll_seconds))
	return bool(path.exists() and path.stat().st_size > 0)


async def _process_job(job: Dict[str, Any]) -> str:
	"""
	Job fields:
	- jump_id: int (required)
	- event_id: int (optional, used for naming)
	- session_id: str (required)
	- clip_start_host: float (required)
	- clip_end_host: float (required)
	- video_fps: int (optional, default 30)
	"""
	jump_id = int(job.get("jump_id") or 0)
	event_id = int(job.get("event_id") or 0)
	session_id = str(job.get("session_id") or "").strip()
	if jump_id <= 0 or not session_id:
		return "invalid job (missing jump_id/session_id)"
	# Enforce strict ownership: if this exact jump_id is gone from DB, do not retry.
	try:
		strict_jump_id = await db.resolve_jump_row_id(jump_id=int(jump_id))
	except Exception:
		strict_jump_id = None
	if not strict_jump_id:
		return "jump row missing in DB (strict jump_id check)"

	clip_start_host = float(job.get("clip_start_host"))
	clip_end_host = float(job.get("clip_end_host"))
	CFG = get_config()
	video_fps = int(job.get("video_fps") or CFG.video.recording_fps)
	sessions_base = Path(CFG.sessions.base_dir)
	jump_clips_subdir = CFG.sessions.jump_clips_subdir.strip()

	base = sessions_base / session_id
	mp4 = base / "video.mp4"
	h264 = base / "video.h264"

	# If the session directory doesn't exist at all, this is a real failure (bad session_id / wrong base_dir).
	if not base.exists():
		return f"session dir not found: {base}"

	# Prefer MP4 if available (fast stream-copy cutting).
	# If MP4 isn't available yet (session still running), attempt a "real-time" clip from the growing H264.
	use_mp4 = bool(mp4.exists() and mp4.stat().st_size > 0)
	use_h264 = bool(h264.exists() and h264.stat().st_size > 0)
	if not use_mp4 and not use_h264:
		return "mp4/h264 not ready yet — retry later"

	# Compute clip offsets using host time directly, anchored by session.json t_start.
	# This removes any dependency on frames.csv or session-level frames in DB.
	t_start = _read_session_t_start(base)
	if t_start is None:
		return "session.json missing t_start (cannot cut clip by host time)"
	start_sec = max(0.0, clip_start_host - float(t_start))
	duration = max(0.2, clip_end_host - clip_start_host)

	clip_backend = get_video_backend_for_tools()

	clips_dir = base / jump_clips_subdir
	clips_dir.mkdir(parents=True, exist_ok=True)
	out_name = f"jump_{event_id}.mp4" if event_id > 0 else f"jump_{jump_id}.mp4"
	out_path = clips_dir / out_name

	# Cut clip:
	# - If MP4 exists: fast stream-copy cut
	# - Else: best-effort cut from raw H264 (may re-encode), enabling "real-time" clips
	ok_cut = False
	if use_mp4:
		try:
			ok_cut = bool(clip_backend.cut_clip_best_effort(mp4, out_path, start_sec=start_sec, duration_sec=duration))
		except Exception:
			ok_cut = False
	else:
		try:
			ok_cut = bool(cut_h264_clip_to_mp4_best_effort(h264, out_path, start_sec=start_sec, duration_sec=duration, fps=video_fps))
		except Exception:
			ok_cut = False

	if not ok_cut:
		return "clip cut failed — retry later"

	# Validate output is a playable MP4 before we publish it to the UI (set video_path).
	# Cutting from a growing H264 can occasionally produce an invalid/truncated MP4.
	try:
		if not (out_path.exists() and out_path.stat().st_size > 2048):
			try:
				if out_path.exists():
					out_path.unlink()
			except Exception:
				pass
			return "clip output too small/absent — retry later"
	except Exception:
		return "clip output stat failed — retry later"

	info = probe_mp4_stream_info(out_path)
	times = extract_mp4_frame_times(out_path)
	if not info or not times:
		# Treat as invalid; delete and retry later.
		try:
			out_path.unlink()
		except Exception:
			pass
		return "clip mp4 not valid/playable yet — retry later"

	# Persist against the strict jump id validated above.
	resolved_jump_id = int(strict_jump_id)

	# Update jump video_path using resolved jump id (preferred)
	rel = str(sessions_base / session_id / jump_clips_subdir / out_name)
	await db.set_jump_video_path_by_jump_id(int(resolved_jump_id), rel)

	# Persist per-jump clip-relative frames (jump_frames) for sync.
	# Derive from the clip MP4 itself (ffprobe) => no frames.csv dependency.
	w = info.get("width")
	h = info.get("height")
	jf = []
	for i, tv in enumerate(times):
		try:
			tv_f = float(tv)
		except Exception:
			continue
		jf.append(
			{
				"frame_idx": int(i),
				"t_video": max(0.0, tv_f),
				"t_host": float(clip_start_host) + max(0.0, tv_f),
				"device_ts": None,
				"width": int(w) if isinstance(w, (int, float)) else None,
				"height": int(h) if isinstance(h, (int, float)) else None,
			}
		)
	await db.replace_jump_frames(int(resolved_jump_id), jf)

	return f"ok (clip={rel}, frames={len(jf)})"


async def _worker_loop(jobs_dir: Path, poll_s: float) -> None:
	# Ensure DB initialized if configured
	try:
		await db.init_db()
	except Exception:
		# DB optional; worker becomes a no-op
		pass

	done_dir = jobs_dir / "done"
	fail_dir = jobs_dir / "failed"
	jobs_dir.mkdir(parents=True, exist_ok=True)
	done_dir.mkdir(parents=True, exist_ok=True)
	fail_dir.mkdir(parents=True, exist_ok=True)
	await _prune_startup_jobs(jobs_dir, fail_dir)

	while True:
		try:
			# Process newest jobs first so fresh jumps get clips quickly even with backlog.
			pending_jobs = list(jobs_dir.glob("*.json"))
			pending_jobs.sort(
				key=lambda p: (p.stat().st_mtime if p.exists() else 0.0),
				reverse=True,
			)
			for path in pending_jobs:
				processing = path.with_suffix(".processing")
				try:
					path.rename(processing)
				except Exception:
					continue
				job = _safe_read_json(processing)
				try:
					msg = await _process_job(job)
					if msg.startswith("ok"):
						processing.rename(done_dir / processing.name.replace(".processing", ".done.json"))
					else:
						# Put back into queue for retry if mp4 not ready; otherwise mark failed.
						if "retry later" in msg:
							processing.rename(path)  # back to pending
						else:
							processing.rename(fail_dir / processing.name.replace(".processing", ".failed.json"))
							_write_text(fail_dir / (processing.stem + ".error.txt"), msg)
				except Exception as e:
					try:
						processing.rename(fail_dir / processing.name.replace(".processing", ".failed.json"))
						_write_text(fail_dir / (processing.stem + ".error.txt"), repr(e))
					except Exception:
						pass
		except Exception:
			pass
		await asyncio.sleep(float(poll_s))


def main(argv: Optional[list[str]] = None) -> int:
	cfg = get_config()
	p = argparse.ArgumentParser(description="Jump clip worker (runs out-of-process)")
	p.add_argument("--jobs-dir", default=cfg.jobs.jump_clip_jobs_dir, help="Directory containing queued job JSON files")
	p.add_argument("--poll-s", type=float, default=float(cfg.clip_worker.poll_seconds), help="Polling interval seconds")
	args = p.parse_args(argv)

	jobs_dir = Path(args.jobs_dir)
	try:
		asyncio.run(_worker_loop(jobs_dir, float(args.poll_s)))
		return 0
	except KeyboardInterrupt:
		return 0
	except Exception as e:
		logger.exception("[jump_clip_worker] fatal: %s", e)
		return 2


if __name__ == "__main__":
	raise SystemExit(main())


