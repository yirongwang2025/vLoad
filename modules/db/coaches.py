"""Coaches: list_coaches, get_coach, upsert_coach, delete_coach, get_coach_skaters."""
from typing import Any, Dict, List, Optional

from modules.db.pool import get_pool

async def list_coaches() -> List[Dict[str, Any]]:
	"""
	List all registered coaches.
	"""
	pool = get_pool()
	if pool is None:
		return []
	async with pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT id, name, email, phone, certification, level, club, notes, created_at, updated_at
			FROM coaches
			ORDER BY name;
			"""
		)
		return [
			{
				"id": int(r["id"]),
				"name": str(r["name"]),
				"email": str(r["email"]) if r["email"] else None,
				"phone": str(r["phone"]) if r["phone"] else None,
				"certification": str(r["certification"]) if r["certification"] else None,
				"level": str(r["level"]) if r["level"] else None,
				"club": str(r["club"]) if r["club"] else None,
				"notes": str(r["notes"]) if r["notes"] else None,
				"created_at": r["created_at"].isoformat() if r["created_at"] else None,
				"updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
			}
			for r in rows
		]


async def get_coach_by_id(coach_id: int) -> Optional[Dict[str, Any]]:
	"""
	Get coach by ID, including assigned skaters.
	"""
	pool = get_pool()
	if pool is None:
		return None
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT id, name, email, phone, certification, level, club, notes, created_at, updated_at
			FROM coaches
			WHERE id = $1;
			""",
			coach_id,
		)
		if not row:
			return None
		coach = {
			"id": int(row["id"]),
			"name": str(row["name"]),
			"email": str(row["email"]) if row["email"] else None,
			"phone": str(row["phone"]) if row["phone"] else None,
			"certification": str(row["certification"]) if row["certification"] else None,
			"level": str(row["level"]) if row["level"] else None,
			"club": str(row["club"]) if row["club"] else None,
			"notes": str(row["notes"]) if row["notes"] else None,
			"created_at": row["created_at"].isoformat() if row["created_at"] else None,
			"updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
		}
		# Include assigned skaters
		coach["skaters"] = await get_coach_skaters(coach_id)
		return coach


async def upsert_coach(
	name: str,
	email: Optional[str] = None,
	phone: Optional[str] = None,
	certification: Optional[str] = None,
	level: Optional[str] = None,
	club: Optional[str] = None,
	notes: Optional[str] = None,
	coach_id: Optional[int] = None,
) -> Dict[str, Any]:
	"""
	Create or update a coach profile.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	name_str = (name or "").strip()
	if not name_str:
		raise ValueError("Coach name is required")
	
	async with pool.acquire() as conn:
		if coach_id:
			# Update existing
			row = await conn.fetchrow(
				"""
				UPDATE coaches
				SET name = $1, email = $2, phone = $3, certification = $4, level = $5,
				    club = $6, notes = $7, updated_at = NOW()
				WHERE id = $8
				RETURNING id, name, email, phone, certification, level, club, notes, created_at, updated_at;
				""",
				name_str,
				email.strip() if email else None,
				phone.strip() if phone else None,
				certification.strip() if certification else None,
				level.strip() if level else None,
				club.strip() if club else None,
				notes.strip() if notes else None,
				coach_id,
			)
		else:
			# Create new
			row = await conn.fetchrow(
				"""
				INSERT INTO coaches (name, email, phone, certification, level, club, notes)
				VALUES ($1, $2, $3, $4, $5, $6, $7)
				RETURNING id, name, email, phone, certification, level, club, notes, created_at, updated_at;
				""",
				name_str,
				email.strip() if email else None,
				phone.strip() if phone else None,
				certification.strip() if certification else None,
				level.strip() if level else None,
				club.strip() if club else None,
				notes.strip() if notes else None,
			)
		if not row:
			raise RuntimeError("Failed to upsert coach")
		return {
			"id": int(row["id"]),
			"name": str(row["name"]),
			"email": str(row["email"]) if row["email"] else None,
			"phone": str(row["phone"]) if row["phone"] else None,
			"certification": str(row["certification"]) if row["certification"] else None,
			"level": str(row["level"]) if row["level"] else None,
			"club": str(row["club"]) if row["club"] else None,
			"notes": str(row["notes"]) if row["notes"] else None,
			"created_at": row["created_at"].isoformat() if row["created_at"] else None,
			"updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
		}


async def delete_coach(coach_id: int) -> Dict[str, Any]:
	"""
	Delete a coach by ID.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			DELETE FROM coaches
			WHERE id = $1
			RETURNING id, name;
			""",
			coach_id,
		)
		if not row:
			return {"deleted": False, "detail": f"No coach found with id={coach_id}"}
		return {
			"deleted": True,
			"id": int(row["id"]),
			"name": str(row["name"]),
		}


async def get_coach_skaters(coach_id: int) -> List[Dict[str, Any]]:
	"""
	Get all skaters assigned to a coach.
	"""
	pool = get_pool()
	if pool is None:
		return []
	async with pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT sc.id, sc.skater_id, sc.is_head_coach, s.name as skater_name
			FROM skater_coaches sc
			JOIN skaters s ON sc.skater_id = s.id
			WHERE sc.coach_id = $1
			ORDER BY sc.is_head_coach DESC, s.name;
			""",
			coach_id,
		)
		return [
			{
				"id": int(r["id"]),
				"skater_id": int(r["skater_id"]),
				"skater_name": str(r["skater_name"]),
				"is_head_coach": bool(r["is_head_coach"]),
			}
			for r in rows
		]


