import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Set, Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Reuse your BLE client
from modules.movesense_gatt import MovesenseGATTClient
from modules.jump_detector import JumpDetectorRealtime

DEVICE = os.getenv("MOVE_DEVICE")  # e.g., "74:92:BA:10:F9:00" or exact name
MODE = os.getenv("MOVE_MODE", "IMU9")  # "IMU6" or "IMU9"
RATE = int(os.getenv("MOVE_RATE", "104"))

# BLE globals
_ble_task: Optional[asyncio.Task] = None
_ble_client: Optional[MovesenseGATTClient] = None

# Jump detection config (Phase 2.5/2.6): tweakable via /config endpoint.
JUMP_CONFIG_DEFAULTS: Dict[str, float] = {
	"min_jump_height_m": 0.15,
	"min_jump_peak_az_no_g": 3.5,
	"min_jump_peak_gz_deg_s": 180.0,
	"min_new_event_separation_s": 0.5,
}
_jump_config: Dict[str, float] = dict(JUMP_CONFIG_DEFAULTS)

# Async queue + worker task for decoupled jump analysis (Option A).
_jump_sample_queue: Optional[asyncio.Queue] = None
_jump_worker_task: Optional[asyncio.Task] = None

# Rolling IMU history for export / offline analysis.
IMU_HISTORY_MAX_SECONDS: float = 60.0
_imu_history: Deque[Dict[str, Any]] = deque()

# Manual gating for jump detection to avoid setup false‑positives.
_jump_detection_enabled: bool = False


def _jump_log_filter(message: str) -> None:
	"""
	Filter JumpDetector log lines before sending them to clients.

	- Always forward Phase 1/2 diagnostics.
	- Only forward [Jump] lines once detection has been manually enabled.
	"""
	global _jump_detection_enabled
	if "[Jump]" in message and not _jump_detection_enabled:
		return
	_log_to_clients(f"[JumpDetector] {message}")


@asynccontextmanager
async def lifespan(app: FastAPI):
	global _ble_task, _ble_client, _jump_sample_queue, _jump_worker_task, _imu_history
	try:
		# Start decoupled jump‑analysis worker and IMU history.
		_jump_sample_queue = asyncio.Queue(maxsize=2000)
		_imu_history = deque()
		_jump_worker_task = asyncio.create_task(_jump_worker_loop())

		yield
	finally:
		# Stop BLE worker
		if _ble_task:
			_ble_task.cancel()
			try:
				await _ble_task
			except asyncio.CancelledError:
				# Expected when cancelling long-running worker
				pass
			except Exception:
				pass
			_ble_task = None

		# Stop jump‑analysis worker
		if _jump_worker_task:
			_jump_worker_task.cancel()
			try:
				await _jump_worker_task
			except asyncio.CancelledError:
				pass
			except Exception:
				pass
		_jump_worker_task = None
		_jump_sample_queue = None

		# Disconnect BLE client
		if _ble_client:
			try:
				await _ble_client.disconnect()
			except Exception:
				pass
			_ble_client = None


app = FastAPI(lifespan=lifespan)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Movesense Live</title>
  <style>
    body { font-family: sans-serif; margin: 16px; }
    #status { margin-bottom: 10px; }
    #data { white-space: pre; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    canvas { border: 1px solid #ddd; margin-top: 10px; }
  </style>
</head>
<body>
  <h2>Movesense Live IMU</h2>

  <div style="margin-top: 8px;">
    <label for="deviceInput"><strong>Device MAC / name:</strong></label>
    <input id="deviceInput" type="text" size="24" placeholder="74:92:BA:10:F9:00" />
    <button id="scanBtn">Scan</button>
    <button id="connectBtn">Connect</button>
    <button id="disconnectBtn">Disconnect</button>
  </div>

  <div style="margin-top: 10px;">
    <strong>Acceleration (X/Y/Z)</strong>
    <canvas id="plotAcc" width="600" height="140" style="display:block; margin-top:4px; border:1px solid #ddd;"></canvas>
  </div>

  <div style="margin-top: 10px;">
    <strong>Gyro (X/Y/Z)</strong>
    <canvas id="plotGyro" width="600" height="140" style="display:block; margin-top:4px; border:1px solid #ddd;"></canvas>
  </div>

  <div style="margin-top: 10px;">
    <strong>Magnetometer (X/Y/Z)</strong>
    <canvas id="plotMag" width="600" height="140" style="display:block; margin-top:4px; border:1px solid #ddd;"></canvas>
  </div>

  <div style="margin-top: 10px; padding: 8px; border: 1px solid #ddd;">
    <strong>Jump detection settings</strong>
    <div style="font-size: 12px; margin-top: 4px;">
      <label>Min height (m):
        <input id="minHeightInput" type="number" step="0.01" style="width:60px;">
      </label>
      <label style="margin-left: 8px;">Min |az-g| (m/s²):
        <input id="minAzInput" type="number" step="0.1" style="width:60px;">
      </label>
      <label style="margin-left: 8px;">Min ωz (°/s):
        <input id="minGzInput" type="number" step="10" style="width:70px;">
      </label>
      <label style="margin-left: 8px;">Min separation (s):
        <input id="minSepInput" type="number" step="0.1" style="width:60px;">
      </label>
      <button id="saveConfigBtn" style="margin-left: 8px;">Save</button>
      <span id="configStatus" style="margin-left: 6px; color:#555;"></span>
    </div>
  </div>

  <div style="margin-top: 10px; padding: 8px; border: 1px solid #ddd;">
    <strong>Jump events</strong>
    <div style="font-size: 12px; margin-top: 4px;">
      <button id="startDetectBtn">Start detection</button>
      <button id="stopDetectBtn" style="margin-left: 6px;">Stop detection</button>
      <span id="jumpStatus" style="margin-left: 8px; color:#555;"></span>
    </div>
    <div style="font-size: 12px; margin-top: 4px;">
      Total jumps: <span id="jumpCount">0</span>
    </div>
    <pre id="jumpList" style="height: 120px; overflow-y: auto; border: 1px solid #ddd; padding: 4px; background: #ffffff; font-size: 11px;"></pre>
  </div>

  <div style="margin-top: 10px;">
    <strong>Log</strong>
    <pre id="logBox" style="height: 160px; overflow-y: auto; border: 1px solid #ddd; padding: 4px; background: #fafafa; font-size: 11px;"></pre>
  </div>
  <script>
    const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
    const deviceInput = document.getElementById('deviceInput');
    const scanBtn = document.getElementById('scanBtn');
    const connectBtn = document.getElementById('connectBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');
    const logBox = document.getElementById('logBox');
    const canvasAcc = document.getElementById('plotAcc');
    const canvasGyro = document.getElementById('plotGyro');
    const canvasMag = document.getElementById('plotMag');
    const ctxAcc = canvasAcc.getContext('2d');
    const ctxGyro = canvasGyro.getContext('2d');
    const ctxMag = canvasMag.getContext('2d');
    const minHeightInput = document.getElementById('minHeightInput');
    const minAzInput = document.getElementById('minAzInput');
    const minGzInput = document.getElementById('minGzInput');
    const minSepInput = document.getElementById('minSepInput');
    const saveConfigBtn = document.getElementById('saveConfigBtn');
    const configStatus = document.getElementById('configStatus');
    const startDetectBtn = document.getElementById('startDetectBtn');
    const stopDetectBtn = document.getElementById('stopDetectBtn');
    const jumpStatus = document.getElementById('jumpStatus');
    const jumpCountEl = document.getElementById('jumpCount');
    const jumpListEl = document.getElementById('jumpList');

    const maxPts = 150;  // number of points kept in history
    // Acceleration series
    const accX = [], accY = [], accZ = [];
    // Gyro series
    const gyroX = [], gyroY = [], gyroZ = [];
    // Magnetometer series
    const magX = [], magY = [], magZ = [];

    let jumpCount = 0;

    const colors = ['#1976d2', '#d32f2f', '#388e3c']; // x=blue, y=red, z=green
    let sampleRate = null; // Hz, from server messages
    let lastDrawTs = 0;    // throttle drawing to avoid overloading CPU

    function addLog(line) {
      const ts = new Date().toISOString();
      logBox.textContent += `[${ts}] ${line}\n`;
      logBox.scrollTop = logBox.scrollHeight;
    }

    function addJump(ev) {
      jumpCount += 1;
      jumpCountEl.textContent = String(jumpCount);

      var tPeak = (typeof ev.t_peak === 'number') ? ev.t_peak.toFixed(3) : '';
      var T_f = (typeof ev.flight_time === 'number') ? ev.flight_time.toFixed(3) : '';
      var h = (typeof ev.height === 'number') ? ev.height.toFixed(3) : '';
      var accPeak = (typeof ev.acc_peak === 'number') ? ev.acc_peak.toFixed(2) : '';
      var gyroPeak = (typeof ev.gyro_peak === 'number') ? ev.gyro_peak.toFixed(0) : '';
      var phase = (typeof ev.rotation_phase === 'number') ? ev.rotation_phase.toFixed(2) : '';
      var conf = (typeof ev.confidence === 'number') ? ev.confidence.toFixed(2) : '';

      var line = 't_peak=' + tPeak + 's, T_f=' + T_f + 's, h=' + h + 'm, ' +
                 'az_no_g=' + accPeak + ', ' +
                 'gz=' + gyroPeak + ', phase=' + phase + ', conf=' + conf;

      var existing = jumpListEl.textContent ? jumpListEl.textContent.split('\\n') : [];
      existing.push(line);
      var maxJumpLines = 50;
      while (existing.length > maxJumpLines) existing.shift();
      jumpListEl.textContent = existing.join('\\n');
    }

    // Restore last-used device from localStorage, if any
    try {
      const last = localStorage.getItem('vload_last_device');
      if (last) {
        deviceInput.value = last;
        addLog(`Restored last device from storage: ${last}`);
      }
    } catch (e) {
      console.error('localStorage error', e);
    }

    // Fetch initial jump config
    async function loadConfig() {
      try {
        const resp = await fetch('/config');
        if (!resp.ok) {
          addLog(`Config load failed with status ${resp.status}`);
          return;
        }
        const data = await resp.json();
        const jc = data.jump || {};
        if (typeof jc.min_jump_height_m === 'number') {
          minHeightInput.value = jc.min_jump_height_m.toFixed(2);
        }
        if (typeof jc.min_jump_peak_az_no_g === 'number') {
          minAzInput.value = jc.min_jump_peak_az_no_g.toFixed(1);
        }
        if (typeof jc.min_jump_peak_gz_deg_s === 'number') {
          minGzInput.value = jc.min_jump_peak_gz_deg_s.toFixed(0);
        }
        if (typeof jc.min_new_event_separation_s === 'number') {
          minSepInput.value = jc.min_new_event_separation_s.toFixed(1);
        }
      } catch (e) {
        console.error(e);
        addLog(`Config load error: ${e}`);
      }
    }

    function pushLimited(arr, value) {
      arr.push(value);
      while (arr.length > maxPts) arr.shift();
    }

    function drawSeries(ctx, series) {
      const canvas = ctx.canvas;
      const h = canvas.height, w = canvas.width;

      const n = Math.max(series[0].length, series[1].length, series[2].length);
      if (n <= 1) {
        ctx.clearRect(0,0,w,h);
        return;
      }

      // Dynamic scale
      let minV = Infinity, maxV = -Infinity;
      series.forEach(arr => {
        arr.forEach(v => {
          if (v < minV) minV = v;
          if (v > maxV) maxV = v;
        });
      });

      if (!isFinite(minV) || !isFinite(maxV)) {
        ctx.clearRect(0,0,w,h);
        return;
      }
      if (maxV === minV) {
        const pad = Math.max(0.5, Math.abs(maxV) * 0.1);
        maxV += pad;
        minV -= pad;
      }

      const range = maxV - minV;
      const padding = 16;

      ctx.clearRect(0,0,w,h);

      // Y labels
      ctx.fillStyle = '#000';
      ctx.font = '10px sans-serif';
      ctx.textBaseline = 'top';
      ctx.fillText(maxV.toFixed(2), 2, 2);
      ctx.textBaseline = 'middle';
      ctx.fillText(((maxV + minV) / 2).toFixed(2), 2, h / 2);
      ctx.textBaseline = 'bottom';
      ctx.fillText(minV.toFixed(2), 2, h - 2);

      // X/time label
      ctx.textBaseline = 'bottom';
      let timeLabel = 'Time';
      if (sampleRate && n > 1) {
        const seconds = n / sampleRate;
        timeLabel += ` (~${seconds.toFixed(1)} s window)`;
      }
      const timeWidth = ctx.measureText(timeLabel).width;
      ctx.fillText(timeLabel, (w - timeWidth) / 2, h - 2);

      // Legend
      const legendX = w - 100;
      const legendY = 4;
      const names = ['X', 'Y', 'Z'];
      ctx.textBaseline = 'top';
      names.forEach((name, idx) => {
        const y = legendY + idx * 12;
        ctx.strokeStyle = colors[idx];
        ctx.beginPath();
        ctx.moveTo(legendX, y + 5);
        ctx.lineTo(legendX + 15, y + 5);
        ctx.stroke();
        ctx.fillStyle = '#000';
        ctx.fillText(name, legendX + 20, y + 1);
      });

      // Plot
      series.forEach((arr, idx) => {
        if (!arr.length) return;
        ctx.strokeStyle = colors[idx];
        ctx.beginPath();
        for (let i = 0; i < arr.length; i++) {
          const x = (i / Math.max(1, maxPts-1)) * (w - 2 * padding) + padding;
          const v = arr[i];
          const norm = (v - minV) / range;
          const y = h - (norm * (h - 2 * padding) + padding);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
    }

    function draw() {
      const now = performance.now();
      // Limit redraws to ~30 FPS to keep UI smooth even at high IMU rates
      if (now - lastDrawTs < 33) {
        requestAnimationFrame(draw);
        return;
      }
      lastDrawTs = now;
      drawSeries(ctxAcc, [accX, accY, accZ]);
      drawSeries(ctxGyro, [gyroX, gyroY, gyroZ]);
      drawSeries(ctxMag, [magX, magY, magZ]);
      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);

    ws.onopen = () => { addLog('WebSocket connected'); loadConfig(); };
    ws.onclose = () => { addLog('WebSocket disconnected'); };
    ws.onerror = () => { addLog('WebSocket error'); };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);

        if (msg.type === 'jump') {
          addJump(msg);
          return;
        }

        // Log-type messages
        if (msg.type === 'log' && typeof msg.msg === 'string') {
          addLog(msg.msg);
          return;
        }

        // IMU data messages
        if (typeof msg.rate === 'number') {
          sampleRate = msg.rate;
        }
        if (msg.first_sample) {
          if (Array.isArray(msg.first_sample.acc)) {
            const a = msg.first_sample.acc;
            pushLimited(accX, a[0] ?? 0);
            pushLimited(accY, a[1] ?? 0);
            pushLimited(accZ, a[2] ?? 0);
          }
          if (Array.isArray(msg.first_sample.gyro)) {
            const g = msg.first_sample.gyro;
            pushLimited(gyroX, g[0] ?? 0);
            pushLimited(gyroY, g[1] ?? 0);
            pushLimited(gyroZ, g[2] ?? 0);
          }
          if (Array.isArray(msg.first_sample.mag)) {
            const m = msg.first_sample.mag;
            pushLimited(magX, m[0] ?? 0);
            pushLimited(magY, m[1] ?? 0);
            pushLimited(magZ, m[2] ?? 0);
          }
        }
      } catch(e) {
        console.error(e);
      }
    };

    // Scan button: call /scan, fill first device into input, and show brief status
    scanBtn.onclick = async () => {
      addLog('Scan button clicked: starting scan...');
      try {
        const resp = await fetch('/scan');
        if (!resp.ok) {
          addLog(`Scan failed with status ${resp.status}`);
          return;
        }
        const data = await resp.json();
        const devices = data.devices || [];
        if (!devices.length) {
          addLog('Scan completed: no Movesense devices found');
          return;
        }
        const first = devices[0];
        deviceInput.value = first.address;
        addLog(`Scan completed: found ${devices.length} device(s); first=${first.name || first.address} (${first.address})`);
      } catch (e) {
        console.error(e);
        addLog(`Scan error: ${e}`);
      }
    };

    // Connect button: POST /connect with the entered device string
    connectBtn.onclick = async () => {
      const device = deviceInput.value.trim();
      if (!device) {
        addLog('Connect clicked with empty device; please enter an address or scan first.');
        return;
      }
      addLog(`Connect request for ${device}`);
      try {
        const resp = await fetch('/connect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device })
        });
        if (!resp.ok) {
          addLog(`Connect failed with status ${resp.status}`);
          return;
        }
        const data = await resp.json();
        addLog(data.detail || 'Connect request sent');

        // Save last-used device
        try {
          localStorage.setItem('vload_last_device', device);
        } catch (e) {
          console.error('localStorage set error', e);
        }
      } catch (e) {
        console.error(e);
        addLog(`Connect error: ${e}`);
      }
    };

    // Disconnect button
    disconnectBtn.onclick = async () => {
      addLog('Disconnect button clicked');
      try {
        const resp = await fetch('/disconnect', { method: 'POST' });
        if (!resp.ok) {
          addLog(`Disconnect failed with status ${resp.status}`);
          return;
        }
        const data = await resp.json();
        addLog(data.detail || 'Disconnect request sent');
      } catch (e) {
        console.error(e);
        addLog(`Disconnect error: ${e}`);
      }
    };

    // Save jump config
    saveConfigBtn.onclick = async () => {
      const payload = {
        min_jump_height_m: parseFloat(minHeightInput.value),
        min_jump_peak_az_no_g: parseFloat(minAzInput.value),
        min_jump_peak_gz_deg_s: parseFloat(minGzInput.value),
        min_new_event_separation_s: parseFloat(minSepInput.value),
      };
      configStatus.textContent = 'Saving...';
      try {
        const resp = await fetch('/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ jump: payload })
        });
        if (!resp.ok) {
          configStatus.textContent = `Save failed (${resp.status})`;
          addLog(`Config save failed with status ${resp.status}`);
          return;
        }
        const data = await resp.json();
        configStatus.textContent = data.detail || 'Saved (reconnect to apply)';
        addLog(data.detail || 'Jump config updated (reconnect to apply)');
      } catch (e) {
        console.error(e);
        configStatus.textContent = 'Save error';
        addLog(`Config save error: ${e}`);
      }
    };

    // Start detection button – enables jump events once the sensor is stable.
    if (startDetectBtn) {
      startDetectBtn.onclick = async () => {
        jumpStatus.textContent = 'Enabling...';
        try {
          const resp = await fetch('/detection/start', { method: 'POST' });
          if (!resp.ok) {
            jumpStatus.textContent = 'Error: ' + resp.status;
            addLog('Detection start failed with status ' + resp.status);
            return;
          }
          const data = await resp.json();
          jumpStatus.textContent = data.detail || 'Detection enabled';
          addLog(data.detail || 'Jump detection enabled');
        } catch (e) {
          console.error(e);
          jumpStatus.textContent = 'Error';
          addLog('Detection start error: ' + e);
        }
      };
    }

    // Stop detection button – disable jump events without disconnecting BLE.
    if (stopDetectBtn) {
      stopDetectBtn.onclick = async () => {
        jumpStatus.textContent = 'Disabling...';
        try {
          const resp = await fetch('/detection/stop', { method: 'POST' });
          if (!resp.ok) {
            jumpStatus.textContent = 'Error: ' + resp.status;
            addLog('Detection stop failed with status ' + resp.status);
            return;
          }
          const data = await resp.json();
          jumpStatus.textContent = data.detail || 'Detection disabled';
          addLog(data.detail || 'Jump detection disabled');
        } catch (e) {
          console.error(e);
          jumpStatus.textContent = 'Error';
          addLog('Detection stop error: ' + e);
        }
      };
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
	return INDEX_HTML

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


def _log_to_clients(message: str) -> None:
	"""
	Send a log line to all connected WebSocket clients.
	Fire-and-forget; safe to call from non-async code.
	"""
	try:
		asyncio.create_task(manager.broadcast_json({"type": "log", "msg": message}))
	except RuntimeError:
		# No running loop yet; ignore
		pass


async def _jump_worker_loop() -> None:
	"""
	Background task that consumes IMU samples from the queue and runs
	JumpDetectorRealtime to emit structured jump events, decoupled from BLE.
	"""
	global _jump_sample_queue

	# One detector instance per worker, using current config snapshot.
	jump_detector = JumpDetectorRealtime(
		sample_rate_hz=RATE,
		window_seconds=3.0,
		logger=_jump_log_filter,
		config=_jump_config,
	)

	while True:
		if _jump_sample_queue is None:
			# Should not happen often; be defensive.
			await asyncio.sleep(0.1)
			continue

		sample = await _jump_sample_queue.get()
		try:
			events = jump_detector.update(sample) or []
		except Exception:
			events = []

		if not events:
			continue

		# Only emit jump messages once detection has been explicitly enabled.
		if not _jump_detection_enabled:
			continue

		for ev in events:
			try:
				jump_msg = {
					"type": "jump",
					"t_peak": ev.get("t_peak"),
					"flight_time": ev.get("flight_time"),
					"height": ev.get("height"),
					"acc_peak": ev.get("peak_az_no_g"),
					"gyro_peak": ev.get("peak_gz"),
					"rotation_phase": ev.get("rotation_phase"),
					"confidence": ev.get("confidence"),
				}
				asyncio.create_task(manager.broadcast_json(jump_msg))
			except Exception:
				# Jump notifications are best‑effort; ignore errors.
				pass

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
	await manager.connect(websocket)
	try:
		while True:
			# Keep connection alive; client doesn't need to send anything
			await websocket.receive_text()
	except WebSocketDisconnect:
		pass
	except Exception:
		pass
	finally:
		await manager.disconnect(websocket)


@app.get("/scan")
async def scan_devices():
	"""
	Scan for nearby Movesense devices and return a list of (address, name).
	"""
	devices = [
		{"address": addr, "name": name}
		async for addr, name in MovesenseGATTClient.scan_for_movesense(timeout=7.0)
	]
	return {"devices": devices}


@app.post("/connect")
async def connect_device(payload: Dict[str, Any]):
	"""
	Start a BLE worker connected to the specified device (MAC or name).
	If a worker is already running, it will be cancelled and replaced.
	"""
	global _ble_task

	try:
		device = (payload.get("device") or "").strip()
		mode = (payload.get("mode") or MODE).strip().upper()
		rate = int(payload.get("rate") or RATE)

		if not device:
			raise HTTPException(status_code=400, detail="Missing 'device' in request body.")

		# Cancel any existing worker
		if _ble_task:
			_ble_task.cancel()
			try:
				await _ble_task
			except asyncio.CancelledError:
				# Normal when restarting worker
				pass
			except Exception:
				pass
			_ble_task = None

		# Start new worker
		_ble_task = asyncio.create_task(ble_worker(device, mode, rate))
		return {"detail": f"Connecting to {device} (mode={mode}, rate={rate})"}
	except HTTPException:
		# Pass through explicit HTTP errors
		raise
	except Exception as e:
		_log_to_clients(f"/connect error: {e!r}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/disconnect")
async def disconnect_device():
	"""
	Stop the current BLE worker (if any) and disconnect the BLE client.
	"""
	global _ble_task, _ble_client

	if _ble_task:
		_ble_task.cancel()
		try:
			await _ble_task
		except asyncio.CancelledError:
			# Expected when stopping worker
			pass
		except Exception:
			pass
		_ble_task = None

	if _ble_client:
		try:
			await _ble_client.disconnect()
		except Exception:
			pass
		_ble_client = None

	_log_to_clients("BLE worker stopped and client disconnected via /disconnect")
	return {"detail": "Disconnected BLE worker and client"}


@app.post("/detection/start")
async def start_jump_detection():
	"""
	Manually enable jump detection, to avoid false positives during sensor
	setup / strapping on.
	"""
	global _jump_detection_enabled
	_jump_detection_enabled = True
	_log_to_clients("[JumpDetector] Detection ENABLED via /detection/start")
	return {"detail": "Jump detection enabled."}


@app.post("/detection/stop")
async def stop_jump_detection():
	"""
	Manually disable jump detection while keeping BLE streaming active.
	"""
	global _jump_detection_enabled
	_jump_detection_enabled = False
	_log_to_clients("[JumpDetector] Detection DISABLED via /detection/stop")
	return {"detail": "Jump detection disabled."}


@app.get("/config")
async def get_config():
	"""
	Return current jump detection configuration.
	"""
	return {"jump": _jump_config}


@app.post("/config")
async def set_config(payload: Dict[str, Any]):
	"""
	Update jump detection configuration. Changes take effect on the next
	connection (when a new JumpDetectorRealtime is created).
	"""
	global _jump_config

	body = payload or {}
	jump_cfg = body.get("jump") or {}
	if not isinstance(jump_cfg, dict):
		raise HTTPException(status_code=400, detail="Field 'jump' must be an object")

	for key, value in jump_cfg.items():
		if key in JUMP_CONFIG_DEFAULTS:
			try:
				_jump_config[key] = float(value)
			except (TypeError, ValueError):
				continue

	return {"detail": "Jump config updated. Reconnect sensor to apply."}

def _compute_analysis(sample: Dict[str, Any]) -> Dict[str, float]:
	# Simple derived metrics for demo
	ax, ay, az = sample["acc"]
	acc_mag = (ax*ax + ay*ay + az*az) ** 0.5
	gx, gy, gz = sample["gyro"]
	gyro_rms = ((gx*gx + gy*gy + gz*gz) / 3.0) ** 0.5
	out = {"acc_mag_first": float(acc_mag), "gyro_rms_first": float(gyro_rms)}
	if "mag" in sample and sample["mag"]:
		mx, my, mz = sample["mag"]
		mag_mag = (mx*mx + my*my + mz*mz) ** 0.5
		out["mag_mag_first"] = float(mag_mag)
	return out

@app.get("/export")
async def export_imu(seconds: float = 30.0):
	"""
	Export recent IMU history for offline analysis / labelling.

	Returns JSON with samples from the last `seconds` (default: 30).
	"""
	now = time.time()
	horizon = max(0.0, float(seconds))
	cutoff = now - horizon

	snap = list(_imu_history)
	samples = [row for row in snap if float(row.get("t", 0.0)) >= cutoff]
	return {"seconds": seconds, "count": len(samples), "samples": samples}


async def ble_worker(device: Optional[str], mode: str, rate: int) -> None:
	global _ble_client, _jump_sample_queue, _imu_history
	print(f"[BLE] Worker starting (device={device!r}, mode={mode}, rate={rate})")
	_log_to_clients(f"BLE worker starting (device={device!r}, mode={mode}, rate={rate})")
	client = MovesenseGATTClient()
	_ble_client = client

	try:
		if device:
			try:
				print(f"[BLE] Connecting to explicit device {device} ...")
				_log_to_clients(f"Attempting explicit connect to {device} ...")
				await client.connect(device)
				_log_to_clients(f"SUCCESS: Connected via explicit device {device}")
			except Exception as e:
				print(f"[BLE] Explicit device connect failed ({e}); falling back to auto-scan.")
				_log_to_clients(f"Explicit connect to {device} failed: {e}. Falling back to auto-scan.")
				device = None  # fall through to scan logic

		if not device:
			# Auto-pick first Movesense found
			print("[BLE] No device specified. Scanning for Movesense devices ...")
			_log_to_clients("No device specified. Scanning for Movesense devices ...")
			found = [item async for item in client.scan_for_movesense(timeout=7.0)]
			print(f"[BLE] Scan complete, found {len(found)} device(s).")
			_log_to_clients(f"Scan complete, found {len(found)} Movesense device(s).")
			if not found:
				print("[BLE] No Movesense devices found during scan.")
				_log_to_clients("ERROR: No Movesense devices found during scan.")
				raise RuntimeError("No Movesense devices found. Ensure the sensor is advertising and nearby.")
			target = found[0]
			print(f"Auto-selecting device: {target[1]} ({target[0]})")
			_log_to_clients(f"Connecting via auto-scan to {target[1]} ({target[0]})")
			await client.connect(target[0])
			_log_to_clients(f"SUCCESS: Connected via auto-scan to {target[1]} ({target[0]})")

		print("[BLE] Connected. Sending HELLO ...")
		_log_to_clients("Connected. Sending HELLO ...")
		await client.hello()
		print("[BLE] HELLO sent. Subscribing to IMU ...")
		_log_to_clients("HELLO sent. Subscribing to IMU stream ...")

		def on_data(payload: bytes) -> None:
			# Try to decode; if decode method not present, fallback to raw hex
			try:
				decoded = client.decode_imu_payload(payload, mode)  # type: ignore[attr-defined]
				samples = decoded.get("samples", [])
				first = samples[0] if samples else None

				if samples:
					now_t = time.time()
					for s in samples:
						# Best‑effort: push to jump‑analysis queue without blocking BLE thread.
						if _jump_sample_queue is not None:
							try:
								_jump_sample_queue.put_nowait(
									{
										"t": now_t,
										"acc": s.get("acc", []),
										"gyro": s.get("gyro", []),
										"mag": s.get("mag", []),
									}
								)
							except Exception:
								# Queue backpressure or other issues: ignore and keep streaming.
								pass

						# Append to rolling IMU history for export
						try:
							_imu_history.append(
								{
									"t": now_t,
									"imu_timestamp": decoded.get("timestamp"),
									"acc": s.get("acc", []),
									"gyro": s.get("gyro", []),
									"mag": s.get("mag", []),
								}
							)
							cutoff = now_t - IMU_HISTORY_MAX_SECONDS
							while _imu_history and float(_imu_history[0].get("t", 0.0)) < cutoff:
								_imu_history.popleft()
						except Exception:
							# History is best‑effort; ignore errors.
							pass

				msg = {
					"t": time.time(),
					"mode": mode,
					"rate": rate,
					"seq": decoded.get("seq"),
					"timestamp": decoded.get("timestamp"),
					"samples_len": len(samples),
					"analysis": _compute_analysis(first) if first else {},
					"first_sample": first,
				}
			except Exception:
				# Log decode problems but keep streaming raw data
				print(f"[BLE] Failed to decode payload of {len(payload)} bytes; sending raw preview.")
				_log_to_clients(f"Decode error on payload of {len(payload)} bytes; sending raw preview.")
				msg = {
					"t": time.time(),
					"mode": mode,
					"rate": rate,
					"raw_len": len(payload),
					"raw_preview": payload.hex()[:96],
				}
			# Fire-and-forget broadcast to WebSocket clients
			asyncio.create_task(manager.broadcast_json(msg))

		sub = await client.subscribe_imu(sample_rate_hz=rate, mode=mode, on_data=on_data)
		print("[BLE] Subscribed to IMU stream; entering sleep loop.")
		_log_to_clients("Subscribed to IMU stream; entering sleep loop.")
		try:
			while True:
				await asyncio.sleep(3600)
		finally:
			print("[BLE] Unsubscribing from IMU ...")
			_log_to_clients("Unsubscribing from IMU ...")
			await client.unsubscribe(sub)
	finally:
		try:
			print("[BLE] Disconnecting client ...")
			_log_to_clients("Disconnecting BLE client ...")
			await _ble_client.disconnect()  # type: ignore
		except Exception:
			pass
		_ble_client = None