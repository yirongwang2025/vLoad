"""
FastAPI dependencies. Use Depends(get_state) in route handlers to receive AppState.
"""
from fastapi import Request

from app_state import AppState


def get_state(request: Request) -> AppState:
	"""Return the app state instance attached in lifespan."""
	return request.app.state.state
