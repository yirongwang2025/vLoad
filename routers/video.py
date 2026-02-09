"""Video backend routes. Routes: /video/connect, disconnect, status, mjpeg, snapshot.jpg, debug."""
import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse

from app_state import AppState
from deps import get_state

router = APIRouter(tags=["video"])


@router.post("/video/connect")
async def video_connect(state: AppState = Depends(get_state)):
	"""Connect to the active video backend and start MJPEG streaming."""
	try:
		state.video.start()
		await asyncio.sleep(0.2)
		st = state.video.get_status()
		if not st.get("running") and st.get("error"):
			err = str(st.get("error"))
			# Camera backend unavailable (e.g. picamera2 on Windows) -> 503 so UI can show a clear message
			if "picamera2" in err.lower() or "ModuleNotFoundError" in err or "not available" in err.lower():
				raise HTTPException(
					status_code=503,
					detail="Camera backend not available on this system. Picamera2 is for Raspberry Pi; on Windows use an HTTP/MJPEG camera or run on a Pi.",
				)
			raise HTTPException(status_code=500, detail=err)
		return {"detail": "Video streaming started.", "status": st}
	except HTTPException:
		raise
	except Exception as e:
		err = str(e)
		if "picamera2" in err.lower() or "ModuleNotFoundError" in err:
			raise HTTPException(
				status_code=503,
				detail="Camera backend not available on this system. Picamera2 is for Raspberry Pi; on Windows use an HTTP/MJPEG camera or run on a Pi.",
			)
		raise HTTPException(status_code=500, detail=f"Video connect failed: {err}")


@router.post("/video/disconnect")
async def video_disconnect(state: AppState = Depends(get_state)):
	"""Stop the active video backend stream."""
	try:
		state.video.stop()
		return {"detail": "Video streaming stopped.", "status": state.video.get_status()}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Video disconnect failed: {e!r}")


@router.get("/video/status")
async def video_status(state: AppState = Depends(get_state)):
	return state.video.get_status()


@router.get("/video/mjpeg")
async def video_mjpeg(fps: float = 15.0, state: AppState = Depends(get_state)):
	"""Live MJPEG stream from the active video backend."""
	return StreamingResponse(
		state.video.mjpeg_stream(fps=float(fps)),
		media_type="multipart/x-mixed-replace; boundary=frame",
		headers={
			"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
			"Pragma": "no-cache",
			"Connection": "keep-alive",
		},
	)


@router.get("/video/snapshot.jpg")
async def video_snapshot(state: AppState = Depends(get_state)):
	"""Return a single latest JPEG frame."""
	jpeg = await state.video.snapshot_jpeg()
	if jpeg is None:
		raise HTTPException(status_code=404, detail="No JPEG frame available yet")
	return Response(
		content=jpeg,
		media_type="image/jpeg",
		headers={
			"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
			"Pragma": "no-cache",
		},
	)


@router.get("/video/debug")
async def video_debug(state: AppState = Depends(get_state)):
	"""Debug info about the latest encoded MJPEG packet."""
	jpeg = await state.video.snapshot_jpeg()
	st = state.video.get_status()
	if jpeg is None:
		return {"status": st, "has_jpeg": False}
	head = jpeg[:8].hex()
	tail = jpeg[-8:].hex() if len(jpeg) >= 8 else jpeg.hex()
	return {
		"status": st,
		"has_jpeg": True,
		"len": len(jpeg),
		"head_hex": head,
		"tail_hex": tail,
		"starts_ffd8": bool(len(jpeg) >= 2 and jpeg[0] == 0xFF and jpeg[1] == 0xD8),
		"ends_ffd9": bool(len(jpeg) >= 2 and jpeg[-2] == 0xFF and jpeg[-1] == 0xD9),
	}
