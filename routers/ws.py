"""WebSocket endpoint and ConnectionManager. Route: /ws."""
import asyncio
import json
from typing import Any, Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["ws"])


class ConnectionManager:
	def __init__(self) -> None:
		self._clients: Set[WebSocket] = set()
		self._lock = asyncio.Lock()

	async def connect(self, websocket: WebSocket) -> None:
		await websocket.accept()
		async with self._lock:
			self._clients.add(websocket)

	async def disconnect(self, websocket: WebSocket) -> None:
		async with self._lock:
			self._clients.discard(websocket)

	async def broadcast_json(self, message: Dict[str, Any]) -> None:
		payload = json.dumps(message, separators=(",", ":"))
		async with self._lock:
			if not self._clients:
				return
			send_tasks = []
			for ws in list(self._clients):
				send_tasks.append(self._send(ws, payload))
			await asyncio.gather(*send_tasks, return_exceptions=True)

	@staticmethod
	async def _send(ws: WebSocket, payload: str) -> None:
		try:
			await ws.send_text(payload)
		except Exception:
			try:
				await ws.close()
			except Exception:
				pass


manager = ConnectionManager()


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
	await manager.connect(websocket)
	try:
		while True:
			await websocket.receive_text()
	except WebSocketDisconnect:
		pass
	except Exception:
		pass
	finally:
		await manager.disconnect(websocket)
