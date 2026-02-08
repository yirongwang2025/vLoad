"""HTML page handlers. No path prefix – routes are /, /jumps, /devices, /skaters, /coaches."""
import base64
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

import state
from modules import db

router = APIRouter(tags=["pages"])


def _get_html(filename: str) -> str:
	"""Load page HTML lazily (B.1). 404 if UI template missing."""
	if state.get_page_html is None:
		raise HTTPException(status_code=503, detail="Server not ready")
	try:
		return state.get_page_html(filename)
	except FileNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e)) from e

_PRELOAD_PLACEHOLDER = "<!-- PRELOAD_SKATERS -->"
_OPTIONS_PLACEHOLDER = "<!-- SKATER_OPTIONS -->"
_PRELOAD_JUMPSS_PLACEHOLDER = "<!-- PRELOAD_JUMPSS -->"
_JUMP_LIST_ITEMS_PLACEHOLDER = "<!-- JUMP_LIST_ITEMS -->"


def _escape_html(s: str) -> str:
	if s is None:
		return ""
	return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


async def _connect_page_html():
	"""Return Connect page HTML with skaters preloaded (shared by GET / and GET /static/index.html)."""
	html = _get_html("index.html")
	placeholder_present = _PRELOAD_PLACEHOLDER in html
	skaters = []
	if _OPTIONS_PLACEHOLDER in html or placeholder_present:
		try:
			skaters = await db.list_skaters()
			if not isinstance(skaters, list):
				skaters = []
		except Exception:
			pass
	# Server-render skater options inside the select so dropdown shows even if client script fails
	if _OPTIONS_PLACEHOLDER in html:
		options_html = "".join(
			f'<option value="{s["id"]}">{_escape_html(s.get("name") or "")}</option>' for s in skaters
		)
		html = html.replace(_OPTIONS_PLACEHOLDER, options_html, 1)
	if placeholder_present:
		json_str = json.dumps(skaters)
		encoded = base64.b64encode(json_str.encode("utf-8")).decode("ascii")
		script = f'<script>window.__PRELOADED_SKATERS__ = JSON.parse(atob("{encoded}"));</script>'
		html = html.replace(_PRELOAD_PLACEHOLDER, script, 1)
	return html


def _connect_page_response(html: str):
	"""Return HTMLResponse with no-cache so browser always gets fresh preloaded skaters."""
	return HTMLResponse(content=html, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


def _shell_response() -> HTMLResponse:
	"""Return SPA shell HTML (B.5). Same for all app routes so client router can mount."""
	html = _get_html("shell.html")
	return HTMLResponse(content=html, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@router.get("/api/fragments/connect", response_class=HTMLResponse)
async def fragment_connect():
	"""Full Connect page HTML for SPA to extract .page (client uses extractPageContent)."""
	return _connect_page_response(await _connect_page_html())


@router.get("/api/fragments/jumps", response_class=HTMLResponse)
async def fragment_jumps():
	"""Full Jump Review page HTML for SPA to extract .page."""
	return _jumps_page_response(await _jumps_page_html())


@router.get("/api/fragments/devices", response_class=HTMLResponse)
async def fragment_devices():
	"""Full devices page HTML for SPA to extract .page."""
	return HTMLResponse(content=_get_html("devices.html"), headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@router.get("/api/fragments/skaters", response_class=HTMLResponse)
async def fragment_skaters():
	"""Full skaters page HTML for SPA to extract .page."""
	return HTMLResponse(content=_get_html("skaters.html"), headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@router.get("/api/fragments/coaches", response_class=HTMLResponse)
async def fragment_coaches():
	"""Full coaches page HTML for SPA to extract .page."""
	return HTMLResponse(content=_get_html("coaches.html"), headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@router.get("/", response_class=HTMLResponse)
async def index():
	"""Serve SPA shell; client loads Connect view from /api/fragments/connect."""
	return _shell_response()


def _jumps_debug_log(obj):
	import os
	try:
		path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cursor", "debug.log")
		with open(path, "a", encoding="utf-8") as f:
			f.write(json.dumps(obj) + "\n")
	except Exception:
		pass


async def _jumps_page_html():
	"""Return Jump Review page HTML with jumps list preloaded (same pattern as Connect/skaters)."""
	# #region agent log
	_jumps_debug_log({"location": "pages.py:_jumps_page_html:entry", "message": "jumps page requested", "data": {}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "hypothesisId": "H1,H2"})
	# #endregion
	html = _get_html("jumps.html")
	placeholder_present = _PRELOAD_JUMPSS_PLACEHOLDER in html
	# #region agent log
	_jumps_debug_log({"location": "pages.py:_jumps_page_html:placeholder", "message": "placeholder check", "data": {"placeholder_present": placeholder_present}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "hypothesisId": "H2"})
	# #endregion
	if not placeholder_present:
		return html
	try:
		jumps_list = await db.list_jumps(limit=200)
		if not isinstance(jumps_list, list):
			jumps_list = []
		# #region agent log
		_jumps_debug_log({"location": "pages.py:_jumps_page_html:after_list", "message": "db.list_jumps result", "data": {"count": len(jumps_list)}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "hypothesisId": "H3,H4"})
		# #endregion
		# Server-render list items so list shows even if script/fetch fails (same pattern as Connect skater options)
		if _JUMP_LIST_ITEMS_PLACEHOLDER in html:
			from datetime import datetime
			items_html_parts = []
			for j in jumps_list[:200]:
				jid = j.get("jump_id") or j.get("id")
				if jid is None:
					continue
				eid = j.get("event_id")
				name = (j.get("name") or "").strip() or (f"Jump {eid}" if eid is not None else "Jump")
				t_peak = j.get("t_peak")
				t_peak_attr = str(t_peak) if t_peak is not None else ""
				eid_attr = str(eid) if eid is not None else ""
				time_label = ""
				if t_peak is not None:
					try:
						dt = datetime.utcfromtimestamp(float(t_peak))
						time_label = dt.strftime("%H:%M:%S")
					except Exception:
						pass
				label_text = _escape_html(name) + (f" ({time_label})" if time_label else "")
				items_html_parts.append(
					f'<li data-jump-id="{jid}" data-event-id="{eid_attr}" data-name="{_escape_html(name)}" data-t-peak="{t_peak_attr}">'
					f'<label>{label_text}</label></li>'
				)
			html = html.replace(_JUMP_LIST_ITEMS_PLACEHOLDER, "".join(items_html_parts), 1)
		json_str = json.dumps(jumps_list)
		encoded = base64.b64encode(json_str.encode("utf-8")).decode("ascii")
		script = f'<script>window.__PRELOADED_JUMPSS__ = JSON.parse(atob("{encoded}"));</script>'
		html = html.replace(_PRELOAD_JUMPSS_PLACEHOLDER, script, 1)
	except Exception as e:
		# #region agent log
		_jumps_debug_log({"location": "pages.py:_jumps_page_html:exception", "message": "preload failed", "data": {"error": str(e)}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "hypothesisId": "H3,H4"})
		# #endregion
		html = html.replace(_PRELOAD_JUMPSS_PLACEHOLDER, "<script>window.__PRELOADED_JUMPSS__ = [];</script>", 1)
	return html


def _jumps_page_response(html: str):
	"""Return HTMLResponse with no-cache so browser gets fresh preloaded jumps."""
	return HTMLResponse(content=html, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@router.get("/jumps", response_class=HTMLResponse)
async def jumps_page():
	"""Serve SPA shell; client loads Jump Review view from /api/fragments/jumps."""
	return _shell_response()


@router.get("/devices", response_class=HTMLResponse)
async def devices_page():
	"""Serve SPA shell; client loads devices view from /api/fragments/devices."""
	return _shell_response()


@router.get("/skaters", response_class=HTMLResponse)
async def skaters_page():
	"""Serve SPA shell; client loads skaters view from /api/fragments/skaters."""
	return _shell_response()


@router.get("/coaches", response_class=HTMLResponse)
async def coaches_page():
	"""Serve SPA shell; client loads coaches view from /api/fragments/coaches."""
	return _shell_response()
