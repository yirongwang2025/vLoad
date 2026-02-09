"""Skaters, skater_devices, skater_coaches, detection settings."""
from typing import Any, Dict, List, Optional

from modules.db.helpers import (
	coach_skater_row_to_dict,
	skater_coach_link_row_to_dict,
	skater_device_link_row_to_dict,
	skater_device_row_to_dict,
	skater_row_to_dict,
)
from modules.db.pool import get_pool

async def list_skaters() -> List[Dict[str, Any]]:
	"""
	List all registered skaters.
	"""
	pool = get_pool()
	if pool is None:
		return []
	async with pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT id, name, date_of_birth, gender, level, club, email, phone, notes, created_at, updated_at
			FROM skaters
			ORDER BY name;
			"""
		)
		return [skater_row_to_dict(r) for r in rows]


async def get_skater_by_id(skater_id: int) -> Optional[Dict[str, Any]]:
	"""
	Get skater by ID, including coaches and devices.
	"""
	pool = get_pool()
	if pool is None:
		return None
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT id, name, date_of_birth, gender, level, club, email, phone, notes, created_at, updated_at
			FROM skaters
			WHERE id = $1;
			""",
			skater_id,
		)
		if not row:
			return None
		skater = skater_row_to_dict(row)
		# Include coaches and devices
		skater["coaches"] = await get_skater_coaches(skater_id)
		skater["devices"] = await get_skater_devices(skater_id)
		return skater


async def upsert_skater(
	name: str,
	date_of_birth: Optional[str] = None,
	gender: Optional[str] = None,
	level: Optional[str] = None,
	club: Optional[str] = None,
	email: Optional[str] = None,
	phone: Optional[str] = None,
	notes: Optional[str] = None,
	skater_id: Optional[int] = None,
) -> Dict[str, Any]:
	"""
	Create or update a skater profile.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	name_str = (name or "").strip()
	if not name_str:
		raise ValueError("Skater name is required")
	
	# Parse date_of_birth if provided
	dob_date = None
	if date_of_birth:
		try:
			from datetime import datetime
			dob_date = datetime.fromisoformat(date_of_birth.replace("Z", "+00:00")).date()
		except Exception:
			pass
	
	async with pool.acquire() as conn:
		if skater_id:
			# Update existing
			row = await conn.fetchrow(
				"""
				UPDATE skaters
				SET name = $1, date_of_birth = $2, gender = $3, level = $4, club = $5,
				    email = $6, phone = $7, notes = $8, updated_at = NOW()
				WHERE id = $9
				RETURNING id, name, date_of_birth, gender, level, club, email, phone, notes, created_at, updated_at;
				""",
				name_str,
				dob_date,
				gender.strip() if gender else None,
				level.strip() if level else None,
				club.strip() if club else None,
				email.strip() if email else None,
				phone.strip() if phone else None,
				notes.strip() if notes else None,
				skater_id,
			)
		else:
			# Create new
			row = await conn.fetchrow(
				"""
				INSERT INTO skaters (name, date_of_birth, gender, level, club, email, phone, notes)
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
				RETURNING id, name, date_of_birth, gender, level, club, email, phone, notes, created_at, updated_at;
				""",
				name_str,
				dob_date,
				gender.strip() if gender else None,
				level.strip() if level else None,
				club.strip() if club else None,
				email.strip() if email else None,
				phone.strip() if phone else None,
				notes.strip() if notes else None,
			)
		if not row:
			raise RuntimeError("Failed to upsert skater")
		return skater_row_to_dict(row)


async def delete_skater(skater_id: int) -> Dict[str, Any]:
	"""
	Delete a skater by ID.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			DELETE FROM skaters
			WHERE id = $1
			RETURNING id, name;
			""",
			skater_id,
		)
		if not row:
			return {"deleted": False, "detail": f"No skater found with id={skater_id}"}
		return {
			"deleted": True,
			"id": int(row["id"]),
			"name": str(row["name"]),
		}


async def get_skater_coaches(skater_id: int) -> List[Dict[str, Any]]:
	"""
	Get all coaches assigned to a skater.
	"""
	pool = get_pool()
	if pool is None:
		return []
	async with pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT sc.id, sc.coach_id, sc.is_head_coach, c.name as coach_name
			FROM skater_coaches sc
			JOIN coaches c ON sc.coach_id = c.id
			WHERE sc.skater_id = $1
			ORDER BY sc.is_head_coach DESC, c.name;
			""",
			skater_id,
		)
		return [coach_skater_row_to_dict(r) for r in rows]


async def add_skater_coach(skater_id: int, coach_id: int, is_head_coach: bool = False) -> Dict[str, Any]:
	"""
	Add a coach to a skater. If is_head_coach is True, unset other head coaches for this skater.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	async with pool.acquire() as conn:
		async with conn.transaction():
			# If setting as head coach, unset other head coaches
			if is_head_coach:
				await conn.execute(
					"UPDATE skater_coaches SET is_head_coach = FALSE WHERE skater_id = $1",
					skater_id
				)
			# Insert or update relationship
			row = await conn.fetchrow(
				"""
				INSERT INTO skater_coaches (skater_id, coach_id, is_head_coach)
				VALUES ($1, $2, $3)
				ON CONFLICT (skater_id, coach_id) DO UPDATE SET is_head_coach = EXCLUDED.is_head_coach
				RETURNING id, skater_id, coach_id, is_head_coach;
				""",
				skater_id,
				coach_id,
				is_head_coach,
			)
			if not row:
				raise RuntimeError("Failed to add skater-coach relationship")
			return skater_coach_link_row_to_dict(row)


async def remove_skater_coach(skater_id: int, coach_id: int) -> bool:
	"""
	Remove a coach from a skater.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	async with pool.acquire() as conn:
		result = await conn.execute(
			"DELETE FROM skater_coaches WHERE skater_id = $1 AND coach_id = $2",
			skater_id,
			coach_id,
		)
		return result == "DELETE 1"


# Skater-Device relationship functions
async def get_skater_devices(skater_id: int) -> List[Dict[str, Any]]:
	"""
	Get all devices assigned to a skater.
	"""
	pool = get_pool()
	if pool is None:
		return []
	async with pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT sd.id, sd.device_id, sd.placement, d.name as device_name, d.mac_address
			FROM skater_devices sd
			JOIN devices d ON sd.device_id = d.id
			WHERE sd.skater_id = $1
			ORDER BY sd.placement, d.name;
			""",
			skater_id,
		)
		return [skater_device_row_to_dict(r) for r in rows]


async def add_skater_device(skater_id: int, device_id: int, placement: str = "waist") -> Dict[str, Any]:
	"""
	Add a device to a skater with placement info.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	placement_str = (placement or "waist").strip().lower()
	if placement_str not in ["waist", "chest", "feet"]:
		placement_str = "waist"
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			INSERT INTO skater_devices (skater_id, device_id, placement)
			VALUES ($1, $2, $3)
			ON CONFLICT (skater_id, device_id) DO UPDATE SET placement = EXCLUDED.placement
			RETURNING id, skater_id, device_id, placement;
			""",
			skater_id,
			device_id,
			placement_str,
		)
		if not row:
			raise RuntimeError("Failed to add skater-device relationship")
		return skater_device_link_row_to_dict(row)


async def remove_skater_device(skater_id: int, device_id: int) -> bool:
	"""
	Remove a device from a skater.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	async with pool.acquire() as conn:
		result = await conn.execute(
			"DELETE FROM skater_devices WHERE skater_id = $1 AND device_id = $2",
			skater_id,
			device_id,
		)
		return result == "DELETE 1"


# Coach-Skater relationship functions (reverse lookup)
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


# Skater detection settings functions
async def get_skater_detection_settings(skater_id: int) -> Optional[Dict[str, Any]]:
	"""
	Get detection settings for a skater. Returns None if not set (use defaults).
	"""
	pool = get_pool()
	if pool is None:
		return None
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT min_jump_height_m, min_jump_peak_az_no_g, min_jump_peak_gz_deg_s,
			       min_new_event_separation_s, min_revs, analysis_interval_s
			FROM skater_detection_settings
			WHERE skater_id = $1;
			""",
			skater_id,
		)
		if not row:
			return None
		return {
			"min_jump_height_m": float(row["min_jump_height_m"]) if row["min_jump_height_m"] is not None else None,
			"min_jump_peak_az_no_g": float(row["min_jump_peak_az_no_g"]) if row["min_jump_peak_az_no_g"] is not None else None,
			"min_jump_peak_gz_deg_s": float(row["min_jump_peak_gz_deg_s"]) if row["min_jump_peak_gz_deg_s"] is not None else None,
			"min_new_event_separation_s": float(row["min_new_event_separation_s"]) if row["min_new_event_separation_s"] is not None else None,
			"min_revs": float(row["min_revs"]) if row["min_revs"] is not None else None,
			"analysis_interval_s": float(row["analysis_interval_s"]) if row["analysis_interval_s"] is not None else None,
		}


async def upsert_skater_detection_settings(
	skater_id: int,
	min_jump_height_m: Optional[float] = None,
	min_jump_peak_az_no_g: Optional[float] = None,
	min_jump_peak_gz_deg_s: Optional[float] = None,
	min_new_event_separation_s: Optional[float] = None,
	min_revs: Optional[float] = None,
	analysis_interval_s: Optional[float] = None,
) -> Dict[str, Any]:
	"""Insert or update detection settings for a skater. Returns the saved settings."""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	async with pool.acquire() as conn:
		await conn.execute(
			"""
			INSERT INTO skater_detection_settings (
				skater_id, min_jump_height_m, min_jump_peak_az_no_g, min_jump_peak_gz_deg_s,
				min_new_event_separation_s, min_revs, analysis_interval_s
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7)
			ON CONFLICT (skater_id) DO UPDATE SET
				min_jump_height_m = COALESCE(EXCLUDED.min_jump_height_m, skater_detection_settings.min_jump_height_m),
				min_jump_peak_az_no_g = COALESCE(EXCLUDED.min_jump_peak_az_no_g, skater_detection_settings.min_jump_peak_az_no_g),
				min_jump_peak_gz_deg_s = COALESCE(EXCLUDED.min_jump_peak_gz_deg_s, skater_detection_settings.min_jump_peak_gz_deg_s),
				min_new_event_separation_s = COALESCE(EXCLUDED.min_new_event_separation_s, skater_detection_settings.min_new_event_separation_s),
				min_revs = COALESCE(EXCLUDED.min_revs, skater_detection_settings.min_revs),
				analysis_interval_s = COALESCE(EXCLUDED.analysis_interval_s, skater_detection_settings.analysis_interval_s),
				updated_at = NOW();
			""",
			skater_id,
			min_jump_height_m,
			min_jump_peak_az_no_g,
			min_jump_peak_gz_deg_s,
			min_new_event_separation_s,
			min_revs,
			analysis_interval_s,
		)
		row = await conn.fetchrow(
			"""
			SELECT min_jump_height_m, min_jump_peak_az_no_g, min_jump_peak_gz_deg_s,
			       min_new_event_separation_s, min_revs, analysis_interval_s
			FROM skater_detection_settings WHERE skater_id = $1;
			""",
			skater_id,
		)
		if not row:
			return {}
		return {
			"min_jump_height_m": float(row["min_jump_height_m"]) if row["min_jump_height_m"] is not None else None,
			"min_jump_peak_az_no_g": float(row["min_jump_peak_az_no_g"]) if row["min_jump_peak_az_no_g"] is not None else None,
			"min_jump_peak_gz_deg_s": float(row["min_jump_peak_gz_deg_s"]) if row["min_jump_peak_gz_deg_s"] is not None else None,
			"min_new_event_separation_s": float(row["min_new_event_separation_s"]) if row["min_new_event_separation_s"] is not None else None,
			"min_revs": float(row["min_revs"]) if row["min_revs"] is not None else None,
			"analysis_interval_s": float(row["analysis_interval_s"]) if row["analysis_interval_s"] is not None else None,
		}

