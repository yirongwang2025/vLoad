"""Device API and scan. Routes: /scan, /api/devices*."""
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from modules import db
from modules.config import get_config
from modules.movesense_gatt import MovesenseGATTClient

logger = logging.getLogger(__name__)
router = APIRouter(tags=["api_devices"])
CFG = get_config()


@router.get("/scan")
async def scan_devices():
	"""
	Scan for nearby Movesense devices and return a list of (address, name).
	"""
	devices = [
		{"address": addr, "name": name}
		async for addr, name in MovesenseGATTClient.scan_for_movesense(timeout=float(CFG.ble.scan_timeout_seconds))
	]
	return {"devices": devices}


@router.get("/api/devices")
async def list_devices_endpoint():
	"""List all registered devices."""
	try:
		devices = await db.list_devices()
		if not isinstance(devices, list):
			devices = []
		return {"devices": devices}
	except Exception as e:
		logger.error("[Devices] Error listing devices: %s", e)
		return {"devices": []}


@router.get("/api/devices/{device_id}")
async def get_device_endpoint(device_id: int):
	"""Get a device by ID."""
	try:
		devices = await db.list_devices()
		device = next((d for d in devices if d["id"] == device_id), None)
		if not device:
			raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
		return device
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/devices")
async def create_device_endpoint(payload: Dict[str, Any]):
	"""Create or update a device mapping. Body: {"mac_address": "...", "name": "..."}"""
	try:
		mac_address = (payload.get("mac_address") or "").strip()
		name = (payload.get("name") or "").strip()
		if not mac_address:
			raise HTTPException(status_code=400, detail="Missing 'mac_address' in request body")
		if not name:
			raise HTTPException(status_code=400, detail="Missing 'name' in request body")
		device = await db.upsert_device(mac_address, name)
		return device
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/devices/{device_id}")
async def update_device_endpoint(device_id: int, payload: Dict[str, Any]):
	"""Update a device mapping. Body: {"mac_address": "...", "name": "..."}"""
	try:
		mac_address = (payload.get("mac_address") or "").strip()
		name = (payload.get("name") or "").strip()
		if not mac_address and not name:
			raise HTTPException(status_code=400, detail="At least one of 'mac_address' or 'name' must be provided")
		devices = await db.list_devices()
		existing = next((d for d in devices if d["id"] == device_id), None)
		if not existing:
			raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
		final_mac = mac_address if mac_address else existing["mac_address"]
		final_name = name if name else existing["name"]
		device = await db.upsert_device(final_mac, final_name)
		return device
	except HTTPException:
		raise
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/devices/{device_id}")
async def delete_device_endpoint(device_id: int):
	"""Delete a device mapping."""
	try:
		result = await db.delete_device(device_id)
		if not result.get("deleted"):
			raise HTTPException(status_code=404, detail=result.get("detail", f"Device {device_id} not found"))
		return result
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
