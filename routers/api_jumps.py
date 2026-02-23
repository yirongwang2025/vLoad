"""Jumps DB API and pose. Routes: /db/jumps* (canonical by jump_id), /db/status, /pose/jumps/*/run."""
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from app_state import AppState
from deps import get_state
from modules import db
from modules.config import get_config
from schemas.requests import BulkDeletePayload, JumpMarksPayload

router = APIRouter(tags=["api_jumps"])
CFG = get_config()


def _remove_file_if_exists(path: Path) -> bool:
	try:
		if path.exists():
			path.unlink()
			return True
	except Exception:
		return False
	return False


def _safe_clip_path(path_str: str) -> Optional[Path]:
	try:
		p = Path(path_str)
		p_resolved = p.resolve()
		data_root = Path("data").resolve()
		if data_root == p_resolved or data_root in p_resolved.parents:
			return p_resolved
	except Exception:
		return None
	return None


def _cleanup_deleted_jump_artifacts(jump_ids: list[int], video_paths: list[str]) -> Dict[str, int]:
	"""
	Best-effort filesystem cleanup after jump delete:
	- remove clip files referenced by deleted rows
	- remove pending/processing clip jobs for deleted jump_ids
	"""
	removed_clips = 0
	removed_jobs = 0
	try:
		jobs_dir = Path(CFG.jobs.jump_clip_jobs_dir)
		done_dir = jobs_dir / "done"
		fail_dir = jobs_dir / "failed"
		for jid in jump_ids:
			patterns = [f"jump_{int(jid)}_*.json", f"jump_{int(jid)}_*.processing"]
			for root in (jobs_dir, done_dir, fail_dir):
				for pat in patterns:
					for job_path in root.glob(pat):
						if _remove_file_if_exists(job_path):
							removed_jobs += 1
	except Exception:
		pass

	for vp in video_paths:
		p = _safe_clip_path(str(vp))
		if p is not None and _remove_file_if_exists(p):
			removed_clips += 1
	return {"removed_jobs": int(removed_jobs), "removed_clips": int(removed_clips)}


def _maybe_schedule_pose_after_marks(state: AppState, out: Dict[str, Any], event_id: int) -> None:
	"""If marks are complete, schedule pose run. Uses event_id (in-memory runner)."""
	if not state.maybe_schedule_pose_for_jump:
		return
	if out.get("t_takeoff_video_t") is None or out.get("t_landing_video_t") is None:
		return
	try:
		state.maybe_schedule_pose_for_jump(int(event_id))
	except Exception:
		pass


# ---- List and status (unchanged) ----
@router.get("/db/jumps")
async def db_list_jumps(limit: Optional[int] = None):
	"""List recent jumps from PostgreSQL (ordered by detection time DESC)."""
	try:
		if limit is None:
			limit = int(CFG.api.jumps_list_limit_default)
		rows = await db.list_jumps(limit=limit)
		return {"count": len(rows), "jumps": rows}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB list_jumps failed: {e!r}")


@router.get("/db/status")
async def db_status():
	"""Quick DB diagnostics endpoint."""
	try:
		return db.get_status()
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB status failed: {e!r}")


# ---- Backward-compat: by_event_id (must be before /db/jumps/{jump_id} so path matches) ----
@router.get("/db/jumps/by_event_id/{event_id}")
async def db_get_jump_by_event_id(event_id: int):
	"""Fetch one jump by event_id (resolves to jump_id). Deprecated: use GET /db/jumps/{jump_id}."""
	jid = await db.resolve_jump_id_from_event_id(int(event_id))
	if jid is None:
		raise HTTPException(status_code=404, detail="Jump not found")
	return await db_get_jump(jid)


@router.delete("/db/jumps/by_event_id/{event_id}")
async def db_delete_jump_by_event_id(event_id: int):
	"""Delete one jump by event_id (resolves to jump_id). Deprecated: use DELETE /db/jumps/{jump_id}."""
	jid = await db.resolve_jump_id_from_event_id(int(event_id))
	if jid is None:
		raise HTTPException(status_code=404, detail="Jump not found")
	return await db_delete_jump(jid)


@router.post("/db/jumps/by_event_id/{event_id}/marks")
async def db_mark_jump_video_by_event_id(event_id: int, payload: JumpMarksPayload):
	"""Store marks by event_id (resolves to jump_id). Deprecated: use POST /db/jumps/{jump_id}/marks."""
	jid = await db.resolve_jump_id_from_event_id(int(event_id))
	if jid is None:
		raise HTTPException(status_code=404, detail="Jump not found")
	return await db_mark_jump_video(jid, payload)


@router.post("/db/jumps/by_event_id/{event_id}/recompute_marked_metrics")
async def db_recompute_marked_metrics_by_event_id(event_id: int):
	"""Recompute metrics by event_id (resolves to jump_id). Deprecated: use POST /db/jumps/{jump_id}/recompute_marked_metrics."""
	jid = await db.resolve_jump_id_from_event_id(int(event_id))
	if jid is None:
		raise HTTPException(status_code=404, detail="Jump not found")
	return await db_recompute_marked_metrics(jid)


@router.post("/db/jumps/by_event_id/{event_id}/pose_metrics")
async def db_set_jump_pose_metrics_by_event_id(event_id: int, payload: Dict[str, Any]):
	"""Store pose metrics by event_id (resolves to jump_id). Deprecated: use POST /db/jumps/{jump_id}/pose_metrics."""
	jid = await db.resolve_jump_id_from_event_id(int(event_id))
	if jid is None:
		raise HTTPException(status_code=404, detail="Jump not found")
	return await db_set_jump_pose_metrics(jid, payload)


@router.post("/pose/jumps/by_event_id/{event_id}/run")
async def pose_run_for_jump_by_event_id(event_id: int, payload: Dict[str, Any] = None, state: AppState = Depends(get_state)):
	"""Run pose by event_id. Deprecated: use POST /pose/jumps/{jump_id}/run."""
	if not state.run_pose_for_jump_best_effort:
		raise HTTPException(status_code=501, detail="Pose runner not available")
	payload = payload or {}
	try:
		max_fps = payload.get("max_fps", CFG.pose.max_fps)
		out = await state.run_pose_for_jump_best_effort(
			int(event_id),
			max_fps=float(max_fps) if max_fps is not None else float(CFG.pose.max_fps),
		)
		if out.get("ok"):
			return out
		raise HTTPException(status_code=400, detail=out.get("error") or "pose run failed")
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Pose run failed: {e!r}")


# ---- Canonical routes by jump_id ----
@router.get("/db/jumps/{jump_id}")
async def db_get_jump(jump_id: int):
	"""Fetch one jump + IMU samples by jumps.id (jump_id). Canonical route."""
	try:
		row = await db.get_jump_with_imu_by_jump_id(int(jump_id))
		if not row:
			raise HTTPException(status_code=404, detail="Jump not found")
		return row
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB get_jump failed: {e!r}")


@router.delete("/db/jumps/{jump_id}")
async def db_delete_jump(jump_id: int):
	"""Delete one jump by jumps.id (jump_id). Canonical route."""
	try:
		out = await db.delete_jump_by_jump_id(int(jump_id))
		if out.get("deleted"):
			jid = int(out.get("jump_id") or int(jump_id))
			vp = str(out.get("video_path") or "").strip()
			clean = _cleanup_deleted_jump_artifacts([jid], [vp] if vp else [])
			out["cleanup"] = clean
		return out
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB delete_jump failed: {e!r}")


@router.post("/db/jumps/bulk_delete")
async def db_bulk_delete_jumps(payload: BulkDeletePayload):
	"""Delete multiple jumps by their jump_id values. Body: { "jump_ids": [1, 2, 3, ...] }"""
	try:
		out = await db.delete_jumps_bulk(payload.jump_ids)
		jump_ids = [int(x) for x in out.get("jump_ids", []) if isinstance(x, (int, str)) and str(x).strip()]
		video_paths = [str(x) for x in out.get("video_paths", []) if str(x).strip()]
		if jump_ids:
			clean = _cleanup_deleted_jump_artifacts(jump_ids, video_paths)
			out["cleanup"] = clean
		return out
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB bulk delete failed: {e!r}")


@router.post("/db/jumps/{jump_id}/marks")
async def db_mark_jump_video(jump_id: int, payload: JumpMarksPayload, state: AppState = Depends(get_state)):
	"""Store video-verified takeoff/landing marks by jump_id. Canonical route."""
	which = (payload.which or payload.kind or "").strip().lower()
	t_host, t_video = payload.t_host, payload.t_video
	try:
		th = float(t_host) if t_host is not None else None
		tv = float(t_video) if t_video is not None else None
	except Exception:
		raise HTTPException(status_code=400, detail="t_host and t_video must be numbers (or null)")
	try:
		out = await db.update_jump_video_mark_by_jump_id(int(jump_id), which=which, t_host=th, t_video=tv)
		if not out.get("ok"):
			raise HTTPException(status_code=400, detail=out.get("error") or "update failed")
		ev = out.get("event_id")
		if ev is not None:
			_maybe_schedule_pose_after_marks(state, out, int(ev))
		return out
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB mark failed: {e!r}")


@router.post("/db/jumps/{jump_id}/recompute_marked_metrics")
async def db_recompute_marked_metrics(jump_id: int, state: AppState = Depends(get_state)):
	"""Recompute IMU-based metrics for a jump by jump_id. Canonical route."""
	try:
		out = await db.recompute_marked_imu_metrics_by_jump_id(int(jump_id))
		if not out.get("ok"):
			raise HTTPException(status_code=400, detail=out.get("error") or "recompute failed")
		ev = out.get("event_id")
		if ev is not None and state.maybe_schedule_pose_for_jump:
			try:
				state.maybe_schedule_pose_for_jump(int(ev))
			except Exception:
				pass
		return out
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB recompute_marked_metrics failed: {e!r}")


@router.post("/db/jumps/{jump_id}/pose_metrics")
async def db_set_jump_pose_metrics(jump_id: int, payload: Dict[str, Any]):
	"""Store video pose analysis metrics by jump_id. Canonical route."""
	try:
		ft = payload.get("flight_time_pose")
		h = payload.get("height_pose")
		rev = payload.get("revolutions_pose")
		meta = payload.get("pose_meta")
		if ft is not None:
			ft = float(ft)
		if h is not None:
			h = float(h)
		if rev is not None:
			rev = float(rev)
		if meta is not None and not isinstance(meta, dict):
			raise HTTPException(status_code=400, detail="pose_meta must be an object (dictionary) or null")
		out = await db.update_jump_pose_metrics_by_jump_id(
			jump_id=int(jump_id),
			flight_time_pose=ft,
			height_pose=h,
			revolutions_pose=rev,
			pose_meta=meta,
		)
		if not out.get("ok"):
			raise HTTPException(status_code=400, detail=out.get("error") or "update failed")
		return out
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB pose_metrics failed: {e!r}")


@router.post("/pose/jumps/{jump_id}/run")
async def pose_run_for_jump(jump_id: int, payload: Dict[str, Any] = None, state: AppState = Depends(get_state)):
	"""Run MediaPipe Pose on the per-jump clip by jump_id. Canonical route."""
	if not state.run_pose_for_jump_best_effort:
		raise HTTPException(status_code=501, detail="Pose runner not available")
	row = await db.get_jump_with_imu_by_jump_id(int(jump_id))
	if not row or row.get("event_id") is None:
		raise HTTPException(status_code=404, detail="Jump not found")
	event_id = int(row["event_id"])
	payload = payload or {}
	try:
		max_fps = payload.get("max_fps", CFG.pose.max_fps)
		out = await state.run_pose_for_jump_best_effort(
			event_id,
			max_fps=float(max_fps) if max_fps is not None else float(CFG.pose.max_fps),
		)
		if out.get("ok"):
			return out
		raise HTTPException(status_code=400, detail=out.get("error") or "pose run failed")
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Pose run failed: {e!r}")


