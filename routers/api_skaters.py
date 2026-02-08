"""Skater API. Routes: /api/skaters* including detection-settings."""
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from modules import db

router = APIRouter(tags=["api_skaters"])

JUMP_CONFIG_DEFAULTS: Dict[str, float] = {
	"min_jump_height_m": 0.15,
	"min_jump_peak_az_no_g": 3.5,
	"min_jump_peak_gz_deg_s": 180.0,
	"min_new_event_separation_s": 0.5,
	"analysis_interval_s": 0.5,
	"min_revs": 0.0,
}


@router.get("/api/skaters")
async def list_skaters_endpoint():
	"""List all registered skaters."""
	try:
		skaters = await db.list_skaters()
		if not isinstance(skaters, list):
			skaters = []
		return {"skaters": skaters}
	except Exception as e:
		print(f"[Skaters] Error listing skaters: {e!r}")
		return {"skaters": []}


@router.get("/api/skaters/{skater_id}")
async def get_skater_endpoint(skater_id: int):
	"""Get a skater by ID."""
	try:
		skater = await db.get_skater_by_id(skater_id)
		if not skater:
			raise HTTPException(status_code=404, detail=f"Skater {skater_id} not found")
		return skater
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/skaters")
async def create_skater_endpoint(payload: Dict[str, Any]):
	"""Create a new skater profile."""
	try:
		name = (payload.get("name") or "").strip()
		if not name:
			raise HTTPException(status_code=400, detail="Missing 'name' in request body")
		skater = await db.upsert_skater(
			name=name,
			date_of_birth=payload.get("date_of_birth"),
			gender=payload.get("gender"),
			level=payload.get("level"),
			club=payload.get("club"),
			email=payload.get("email"),
			phone=payload.get("phone"),
			notes=payload.get("notes"),
		)
		return skater
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/skaters/{skater_id}")
async def update_skater_endpoint(skater_id: int, payload: Dict[str, Any]):
	"""Update a skater profile."""
	try:
		name = (payload.get("name") or "").strip()
		if not name:
			raise HTTPException(status_code=400, detail="Missing 'name' in request body")
		skater = await db.upsert_skater(
			name=name,
			date_of_birth=payload.get("date_of_birth"),
			gender=payload.get("gender"),
			level=payload.get("level"),
			club=payload.get("club"),
			email=payload.get("email"),
			phone=payload.get("phone"),
			notes=payload.get("notes"),
			skater_id=skater_id,
		)
		return skater
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/skaters/{skater_id}")
async def delete_skater_endpoint(skater_id: int):
	"""Delete a skater profile."""
	try:
		result = await db.delete_skater(skater_id)
		if not result.get("deleted"):
			raise HTTPException(status_code=404, detail=result.get("detail", f"Skater {skater_id} not found"))
		return result
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/skaters/{skater_id}/detection-settings")
async def get_skater_detection_settings_endpoint(skater_id: int):
	"""Get detection settings for a skater. Returns defaults if not set."""
	try:
		settings = await db.get_skater_detection_settings(skater_id)
		if settings:
			return {"settings": settings}
		return {"settings": None, "defaults": JUMP_CONFIG_DEFAULTS}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/skaters/{skater_id}/detection-settings")
async def set_skater_detection_settings_endpoint(skater_id: int, payload: Dict[str, Any]):
	"""Save detection settings for a skater."""
	try:
		jump_cfg = payload.get("jump") or payload
		if not isinstance(jump_cfg, dict):
			raise HTTPException(status_code=400, detail="Settings must be an object")
		result = await db.upsert_skater_detection_settings(
			skater_id=skater_id,
			min_jump_height_m=jump_cfg.get("min_jump_height_m"),
			min_jump_peak_az_no_g=jump_cfg.get("min_jump_peak_az_no_g"),
			min_jump_peak_gz_deg_s=jump_cfg.get("min_jump_peak_gz_deg_s"),
			min_new_event_separation_s=jump_cfg.get("min_new_event_separation_s"),
			min_revs=jump_cfg.get("min_revs"),
			analysis_interval_s=jump_cfg.get("analysis_interval_s"),
		)
		return {"detail": "Detection settings saved for skater", "settings": result}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
