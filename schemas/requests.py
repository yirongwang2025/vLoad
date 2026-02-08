"""Pydantic request body models for high-traffic endpoints."""
from typing import List, Optional

from pydantic import BaseModel, Field


class ConnectPayload(BaseModel):
	"""Request body for POST /connect. Start IMU collector by device or skater_id."""

	device: Optional[str] = Field(None, description="MAC address or registered device name")
	skater_id: Optional[int] = Field(None, description="Use first registered device for this skater")
	mode: Optional[str] = Field(None, description="IMU mode, e.g. IMU6 or IMU9")
	rate: Optional[int] = Field(None, description="Sample rate in Hz, e.g. 104")


class SessionStartPayload(BaseModel):
	"""Request body for POST /session/start. Optional session_id; auto-generated if omitted."""

	session_id: Optional[str] = Field(None, description="Session identifier; generated if empty")


class JumpMarksPayload(BaseModel):
	"""Request body for POST /db/jumps/.../marks. Video-verified takeoff/landing mark."""

	which: Optional[str] = Field(None, description="'start' (takeoff) or 'end' (landing)")
	kind: Optional[str] = Field(None, description="Alias for 'which'")
	t_host: Optional[float] = Field(None, description="Host timestamp (epoch seconds)")
	t_video: Optional[float] = Field(None, description="Video timeline time (seconds)")


class BulkDeletePayload(BaseModel):
	"""Request body for POST /db/jumps/bulk_delete. List of jump_id values to delete."""

	jump_ids: List[int] = Field(..., min_length=1, description="Jump IDs to delete")
