import asyncio
import json
import os
import time
from typing import Set, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Reuse your BLE client
from movesense_gatt import MovesenseGATTClient

DEVICE = os.getenv("MOVE_DEVICE")  # e.g., "74:92:BA:10:F9:00" or exact name
MODE = os.getenv("MOVE_MODE", "IMU9")  # "IMU6" or "IMU9"
RATE = int(os.getenv("MOVE_RATE", "104"))

# BLE globals
_ble_task: Optional[asyncio.Task] = None
_ble_client: Optional[MovesenseGATTClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
	global _ble_task, _ble_client
	_ble_task = asyncio.create_task(ble_worker(DEVICE, MODE, RATE))
	try:
		yield
	finally:
		if _ble_task:
			_ble_task.cancel()
			try:
				await _ble_task
			except Exception:
				pass
			_ble_task = None
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
  <div id="status">Connecting...</div>

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
  <script>
    const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
    const status = document.getElementById('status');
    const canvasAcc = document.getElementById('plotAcc');
    const canvasGyro = document.getElementById('plotGyro');
    const canvasMag = document.getElementById('plotMag');
    const ctxAcc = canvasAcc.getContext('2d');
    const ctxGyro = canvasGyro.getContext('2d');
    const ctxMag = canvasMag.getContext('2d');

    const maxPts = 300;
    // Acceleration series
    const accX = [], accY = [], accZ = [];
    // Gyro series
    const gyroX = [], gyroY = [], gyroZ = [];
    // Magnetometer series
    const magX = [], magY = [], magZ = [];

    const colors = ['#1976d2', '#d32f2f', '#388e3c']; // x=blue, y=red, z=green
    let sampleRate = null; // Hz, from server messages

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
      drawSeries(ctxAcc, [accX, accY, accZ]);
      drawSeries(ctxGyro, [gyroX, gyroY, gyroZ]);
      drawSeries(ctxMag, [magX, magY, magZ]);
      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);

    ws.onopen = () => { status.textContent = 'Connected'; };
    ws.onclose = () => { status.textContent = 'Disconnected'; };
    ws.onerror = () => { status.textContent = 'Error'; };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
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

async def ble_worker(device: Optional[str], mode: str, rate: int) -> None:
	global _ble_client
	print(f"[BLE] Worker starting (device={device!r}, mode={mode}, rate={rate})")
	client = MovesenseGATTClient()
	_ble_client = client

	try:
		if device:
			print(f"[BLE] Connecting to explicit device {device} ...")
			await client.connect(device)
		else:
			# Auto-pick first Movesense found
			print("[BLE] No device specified. Scanning for Movesense devices ...")
			found = [item async for item in client.scan_for_movesense(timeout=7.0)]
			print(f"[BLE] Scan complete, found {len(found)} device(s).")
			if not found:
				print("[BLE] No Movesense devices found during scan.")
				raise RuntimeError("No Movesense devices found. Ensure the sensor is advertising and nearby.")
			target = found[0]
			print(f"Auto-selecting device: {target[1]} ({target[0]})")
			await client.connect(target[0])

		print("[BLE] Connected. Sending HELLO ...")
		await client.hello()
		print("[BLE] HELLO sent. Subscribing to IMU ...")

		def on_data(payload: bytes) -> None:
			# Try to decode; if decode method not present, fallback to raw hex
			try:
				decoded = client.decode_imu_payload(payload, mode)  # type: ignore[attr-defined]
				samples = decoded.get("samples", [])
				first = samples[0] if samples else None
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
		try:
			while True:
				await asyncio.sleep(3600)
		finally:
			print("[BLE] Unsubscribing from IMU ...")
			await client.unsubscribe(sub)
	finally:
		try:
			print("[BLE] Disconnecting client ...")
			await _ble_client.disconnect()  # type: ignore
		except Exception:
			pass
		_ble_client = None