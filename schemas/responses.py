"""Pydantic response models for API docs (optional; routes may return dicts)."""
from typing import Optional

from pydantic import BaseModel


class ConnectResponse(BaseModel):
	"""Response from POST /connect."""

	detail: str
	collector_pid: Optional[int] = None


class SessionStartResponse(BaseModel):
	"""Response from POST /session/start."""

	detail: str
	session_id: Optional[str] = None
	dir: Optional[str] = None
