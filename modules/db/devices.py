"""Devices: list_devices, get_device, upsert_device, delete_device, resolve_device_identifier."""
from typing import Any, Dict, List, Optional

from modules.db.helpers import device_row_to_dict
from modules.db.pool import get_pool

# Device management functions
async def list_devices() -> List[Dict[str, Any]]:
	"""
	List all registered devices.
	"""
	pool = get_pool()
	if pool is None:
		return []
	async with pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT id, mac_address, name, created_at, updated_at
			FROM devices
			ORDER BY name, mac_address;
			"""
		)
		return [device_row_to_dict(r) for r in rows]


async def get_device_by_mac(mac_address: str) -> Optional[Dict[str, Any]]:
	"""
	Get device by MAC address.
	"""
	pool = get_pool()
	if pool is None:
		return None
	mac = (mac_address or "").strip().upper()
	if not mac:
		return None
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT id, mac_address, name, created_at, updated_at
			FROM devices
			WHERE mac_address = $1;
			""",
			mac,
		)
		if not row:
			return None
		return device_row_to_dict(row)


async def get_device_by_name(name: str) -> Optional[Dict[str, Any]]:
	"""
	Get device by name.
	"""
	pool = get_pool()
	if pool is None:
		return None
	name_str = (name or "").strip()
	if not name_str:
		return None
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT id, mac_address, name, created_at, updated_at
			FROM devices
			WHERE name = $1;
			""",
			name_str,
		)
		if not row:
			return None
		return device_row_to_dict(row)


async def upsert_device(mac_address: str, name: str) -> Dict[str, Any]:
	"""
	Create or update a device mapping. Returns the device record.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	mac = (mac_address or "").strip().upper()
	name_str = (name or "").strip()
	if not mac:
		raise ValueError("MAC address is required")
	if not name_str:
		raise ValueError("Device name is required")
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			INSERT INTO devices (mac_address, name)
			VALUES ($1, $2)
			ON CONFLICT (mac_address) DO UPDATE SET
				name = EXCLUDED.name,
				updated_at = NOW()
			RETURNING id, mac_address, name, created_at, updated_at;
			""",
			mac,
			name_str,
		)
		if not row:
			raise RuntimeError("Failed to upsert device")
		return device_row_to_dict(row)


async def delete_device(device_id: int) -> Dict[str, Any]:
	"""
	Delete a device by ID.
	"""
	pool = get_pool()
	if pool is None:
		raise RuntimeError("Database pool not initialized")
	async with pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			DELETE FROM devices
			WHERE id = $1
			RETURNING id, mac_address, name;
			""",
			device_id,
		)
		if not row:
			return {"deleted": False, "detail": f"No device found with id={device_id}"}
		return {
			"deleted": True,
			"id": int(row["id"]),
			"mac_address": str(row["mac_address"]),
			"name": str(row["name"]),
		}


async def resolve_device_identifier(identifier: str) -> Optional[str]:
	"""
	Resolve a device identifier (MAC address or name) to MAC address.
	Returns the MAC address if found, None otherwise.
	"""
	pool = get_pool()
	if pool is None:
		return None
	identifier_str = (identifier or "").strip()
	if not identifier_str:
		return None
	
	# If it looks like a MAC address, normalize and check
	if ":" in identifier_str or len(identifier_str) == 17:
		mac = identifier_str.upper()
		device = await get_device_by_mac(mac)
		if device:
			return device["mac_address"]
		# Also return the MAC as-is if it's valid format (might not be in DB yet)
		return mac
	
	# Otherwise, treat as name and look up
	device = await get_device_by_name(identifier_str)
	if device:
		return device["mac_address"]
	
	return None


