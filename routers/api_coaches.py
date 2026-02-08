"""Coach API and skater–coach link routes. Routes: /api/coaches*, /api/skaters/*/coaches, /api/skaters/*/devices, /api/coaches/*/skaters."""
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from modules import db

router = APIRouter(tags=["api_coaches"])


@router.get("/api/coaches")
async def list_coaches_endpoint():
	"""List all registered coaches."""
	try:
		coaches = await db.list_coaches()
		if not isinstance(coaches, list):
			coaches = []
		return {"coaches": coaches}
	except Exception as e:
		print(f"[Coaches] Error listing coaches: {e!r}")
		return {"coaches": []}


@router.get("/api/coaches/{coach_id}")
async def get_coach_endpoint(coach_id: int):
	"""Get a coach by ID."""
	try:
		coach = await db.get_coach_by_id(coach_id)
		if not coach:
			raise HTTPException(status_code=404, detail=f"Coach {coach_id} not found")
		return coach
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/coaches")
async def create_coach_endpoint(payload: Dict[str, Any]):
	"""Create a new coach profile."""
	try:
		name = (payload.get("name") or "").strip()
		if not name:
			raise HTTPException(status_code=400, detail="Missing 'name' in request body")
		coach = await db.upsert_coach(
			name=name,
			email=payload.get("email"),
			phone=payload.get("phone"),
			certification=payload.get("certification"),
			level=payload.get("level"),
			club=payload.get("club"),
			notes=payload.get("notes"),
		)
		return coach
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/coaches/{coach_id}")
async def update_coach_endpoint(coach_id: int, payload: Dict[str, Any]):
	"""Update a coach profile."""
	try:
		name = (payload.get("name") or "").strip()
		if not name:
			raise HTTPException(status_code=400, detail="Missing 'name' in request body")
		coach = await db.upsert_coach(
			name=name,
			email=payload.get("email"),
			phone=payload.get("phone"),
			certification=payload.get("certification"),
			level=payload.get("level"),
			club=payload.get("club"),
			notes=payload.get("notes"),
			coach_id=coach_id,
		)
		return coach
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/coaches/{coach_id}")
async def delete_coach_endpoint(coach_id: int):
	"""Delete a coach profile."""
	try:
		result = await db.delete_coach(coach_id)
		if not result.get("deleted"):
			raise HTTPException(status_code=404, detail=result.get("detail", f"Coach {coach_id} not found"))
		return result
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/skaters/{skater_id}/coaches")
async def add_skater_coach_endpoint(skater_id: int, payload: Dict[str, Any]):
	"""Add a coach to a skater. Body: {"coach_id": int, "is_head_coach": bool}"""
	try:
		coach_id = int(payload.get("coach_id") or 0)
		is_head_coach = bool(payload.get("is_head_coach", False))
		if coach_id <= 0:
			raise HTTPException(status_code=400, detail="Missing or invalid 'coach_id'")
		result = await db.add_skater_coach(skater_id, coach_id, is_head_coach)
		return result
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/skaters/{skater_id}/coaches/{coach_id}")
async def remove_skater_coach_endpoint(skater_id: int, coach_id: int):
	"""Remove a coach from a skater."""
	try:
		success = await db.remove_skater_coach(skater_id, coach_id)
		if not success:
			raise HTTPException(status_code=404, detail="Relationship not found")
		return {"deleted": True}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/skaters/{skater_id}/devices")
async def add_skater_device_endpoint(skater_id: int, payload: Dict[str, Any]):
	"""Add a device to a skater. Body: {"device_id": int, "placement": "waist"|"chest"|"feet"}"""
	try:
		device_id = int(payload.get("device_id") or 0)
		placement = (payload.get("placement") or "waist").strip().lower()
		if device_id <= 0:
			raise HTTPException(status_code=400, detail="Missing or invalid 'device_id'")
		if placement not in ["waist", "chest", "feet"]:
			placement = "waist"
		result = await db.add_skater_device(skater_id, device_id, placement)
		return result
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/skaters/{skater_id}/devices/{device_id}")
async def remove_skater_device_endpoint(skater_id: int, device_id: int):
	"""Remove a device from a skater."""
	try:
		success = await db.remove_skater_device(skater_id, device_id)
		if not success:
			raise HTTPException(status_code=404, detail="Relationship not found")
		return {"deleted": True}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/coaches/{coach_id}/skaters")
async def add_coach_skater_endpoint(coach_id: int, payload: Dict[str, Any]):
	"""Add a skater to a coach. Body: {"skater_id": int, "is_head_coach": bool}"""
	try:
		skater_id = int(payload.get("skater_id") or 0)
		is_head_coach = bool(payload.get("is_head_coach", False))
		if skater_id <= 0:
			raise HTTPException(status_code=400, detail="Missing or invalid 'skater_id'")
		result = await db.add_skater_coach(skater_id, coach_id, is_head_coach)
		return result
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/coaches/{coach_id}/skaters/{skater_id}")
async def remove_coach_skater_endpoint(coach_id: int, skater_id: int):
	"""Remove a skater from a coach."""
	try:
		success = await db.remove_skater_coach(skater_id, coach_id)
		if not success:
			raise HTTPException(status_code=404, detail="Relationship not found")
		return {"deleted": True}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
