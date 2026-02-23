from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import time

logger = logging.getLogger(__name__)
from collections import deque
from typing import Any, Dict, Optional, Tuple

from modules.config import get_config
from modules.movesense_gatt import MovesenseGATTClient


def _json_dumps(obj: Any) -> bytes:
	# Compact JSON for UDP.
	return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


async def _run_collector(
	device: str,
	mode: str,
	rate: int,
	udp_host: str,
	udp_port: int,
) -> None:
	"""
	Standalone BLE collector process:
	- connects to Movesense via Bleak
	- subscribes IMU stream
	- assigns per-sample timestamps (monotonic wall-clock)
	- ships packets to server over localhost UDP

	Why UDP?
	- minimal overhead, no backpressure into BLE callback
	- process isolation (own event loop + GIL) from video/server load
	"""
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.setblocking(False)
	target = (udp_host, int(udp_port))
	cfg = get_config()
	ccfg = cfg.collector
	rcfg = cfg.runtime

	def send(obj: Dict[str, Any]) -> None:
		try:
			sock.sendto(_json_dumps(obj), target)
		except Exception:
			# Best-effort: don't crash the collector due to transient UDP send errors.
			pass

	disconnects: int = 0
	last_disconnect_t: Optional[float] = None
	last_error: Optional[str] = None
	rx_window = deque()  # (t_mono, n_samples)
	rx_pkts = deque()  # (t_mono, 1)
	last_notify_mono = time.monotonic()

	def send_stat() -> None:
		# Collector-side rolling 5s receive rate (BLE ground truth, independent of server/WS).
		try:
			now_m = time.monotonic()
			cut = now_m - float(rcfg.imu_rx_rate_window_seconds)
			while rx_window and rx_window[0][0] < cut:
				rx_window.popleft()
			while rx_pkts and rx_pkts[0][0] < cut:
				rx_pkts.popleft()
			s5 = sum(n for _, n in rx_window)
			p5 = len(rx_pkts)
			rate5 = float(s5) / float(rcfg.imu_rx_rate_window_seconds)
			stale_s = max(0.0, now_m - last_notify_mono)
		except Exception:
			s5, p5, rate5, stale_s = None, None, None, None
		send(
			{
				"type": "collector_stat",
				"disconnects": disconnects,
				"last_disconnect_t": last_disconnect_t,
				"last_error": last_error,
				"rx_packets_5s": p5,
				"rx_samples_5s": s5,
				"rx_rate_hz_5s": rate5,
				"notify_stale_s": stale_s,
			}
		)

	def mk_client() -> MovesenseGATTClient:
		c = MovesenseGATTClient(logger=lambda s: send({"type": "log", "msg": str(s)}))

		def _on_disc() -> None:
			nonlocal disconnects, last_disconnect_t
			disconnects += 1
			last_disconnect_t = time.time()
			send_stat()

		c.set_disconnected_callback(_on_disc)
		return c

	client = mk_client()
	connected_addr: Optional[str] = None

	# Buffer for raw notifications; processing happens on this process's loop.
	# IMPORTANT: timestamp in the BLE callback thread so IMU times are not skewed by
	# collector processing backlog (which can be seconds under video load).
	payload_q: asyncio.Queue[Tuple[float, bytes]] = asyncio.Queue(maxsize=int(ccfg.payload_queue_maxsize))
	last_notify_mono = time.monotonic()

	def on_data(payload: bytes) -> None:
		# Keep callback minimal; Bleak may invoke from another thread.
		nonlocal last_notify_mono
		last_notify_mono = time.monotonic()
		t_cb = time.time()
		# Bleak may invoke this on a non-async thread. We capture the main loop once.
		try:
			main_loop.call_soon_threadsafe(payload_q.put_nowait, (t_cb, payload))
		except Exception:
			# Drop if queue is full or loop is closing.
			pass

	# Per-sample timestamping state
	last_sample_t = 0.0
	# Device timestamp (ms) -> epoch seconds mapping:
	#   epoch_s ≈ device_ms/1000 + offset_s
	#
	# Proposed approach: compute a fixed offset once at connect time by averaging
	# ~1s of packets, then reuse it for the rest of the connection. This reduces
	# jitter from per-packet BLE scheduling and avoids "backlog skew" entirely.
	CALIB_WARMUP_S: float = float(ccfg.calib_warmup_seconds)
	CALIB_MIN_SAMPLES: int = int(ccfg.calib_min_samples)
	_offset_fixed_s: Optional[float] = None
	_offset_samples: list[float] = []
	_calib_start_mono: Optional[float] = None
	# If we haven't finished calibration yet, we fall back to a lightweight EMA.
	offset_ema_s: Optional[float] = None
	offset_alpha: float = float(ccfg.offset_alpha)

	async def process_payloads() -> None:
		nonlocal last_sample_t
		nonlocal offset_ema_s
		nonlocal _offset_fixed_s, _offset_samples, _calib_start_mono
		last_stat_mono = time.monotonic()
		while True:
			t_cb, payload = await payload_q.get()
			try:
				decoded = client.decode_imu_payload(payload, mode)  # type: ignore[attr-defined]
				samples = decoded.get("samples", []) or []
				frame_ts = decoded.get("timestamp")
				seq = decoded.get("seq")
				if not samples:
					continue

				n = len(samples)
				dt = 1.0 / float(rate) if rate and rate > 0 else 1.0 / 104.0
				dt_ms = dt * 1000.0

				# Prefer device timestamp (ms) if available; it's more accurate for intra-packet timing.
				use_device_ts = isinstance(frame_ts, (int, float)) and float(frame_ts) > 0

				# Determine an epoch anchor for this packet:
				# - If we have device timestamp: align the LAST sample time to callback receipt time.
				# - Else: fall back to callback time as the end-of-packet timestamp.
				now_frame_t = float(t_cb)
				frame_ts_ms = float(frame_ts) if use_device_ts else None
				if frame_ts_ms is not None:
					# Device timestamp for the last sample in this packet:
					device_last_s = (frame_ts_ms + float(max(0, n - 1)) * dt_ms) / 1000.0
					offset_inst = now_frame_t - device_last_s
					# Start calibration window on first device-timestamped packet.
					if _calib_start_mono is None:
						_calib_start_mono = time.monotonic()
					# If we don't have a fixed offset yet, collect samples during warmup.
					if _offset_fixed_s is None and _calib_start_mono is not None:
						try:
							if (time.monotonic() - _calib_start_mono) <= CALIB_WARMUP_S:
								_offset_samples.append(float(offset_inst))
							else:
								# Finalize: use median for robustness to BLE jitter spikes.
								if len(_offset_samples) >= CALIB_MIN_SAMPLES:
									s = sorted(_offset_samples)
									_offset_fixed_s = float(s[len(s) // 2])
								else:
									# Not enough samples; fall back to EMA.
									_offset_fixed_s = None
						except Exception:
							pass
					# Maintain an EMA until fixed offset is established (and also as a debug signal).
					if offset_ema_s is None:
						offset_ema_s = float(offset_inst)
					else:
						offset_ema_s = float(offset_ema_s) * (1.0 - offset_alpha) + float(offset_inst) * offset_alpha

				out_samples = []
				for i, s in enumerate(samples):
					# Compute per-sample epoch time.
					# - If device timestamp is present: use calibrated mapping device_ms -> epoch_s.
					# - Else: distribute timestamps across the packet ending at now_frame_t.
					if frame_ts_ms is not None and (_offset_fixed_s is not None or offset_ema_s is not None):
						device_ms = frame_ts_ms + float(i) * dt_ms
						device_ts_s = device_ms / 1000.0
						offset_use = _offset_fixed_s if _offset_fixed_s is not None else offset_ema_s
						t_i = float(device_ts_s) + float(offset_use)
					else:
						t_i = now_frame_t - float((n - 1 - i)) * dt
						device_ts_s = None

					# Enforce monotonicity (protect downstream jump detector).
					if t_i <= last_sample_t:
						t_i = last_sample_t + dt
					last_sample_t = t_i
					out_samples.append(
						{
							"t": t_i,
							"acc": s.get("acc", []),
							"gyro": s.get("gyro", []),
							"mag": s.get("mag", []),
							"imu_timestamp": frame_ts,
							"imu_sample_index": i,
							# Optional: device timestamp in seconds (non-epoch; "since boot" timebase)
							"device_ts": device_ts_s,
						}
					)

				send(
					{
						"type": "imu",
						"mode": mode,
						"rate": rate,
						"seq": seq,
						"timestamp": frame_ts,
						"samples": out_samples,
						"t_host": now_frame_t,
						# Debug: current best estimate(s) of epoch offset for device timestamp mapping.
						"imu_clock_offset_s": (_offset_fixed_s if _offset_fixed_s is not None else offset_ema_s),
						"imu_clock_offset_fixed": bool(_offset_fixed_s is not None),
						"imu_clock_offset_calib_n": int(len(_offset_samples)),
					}
				)

				# Update collector rx rate window (after successful decode)
				try:
					now_m = time.monotonic()
					rx_window.append((now_m, int(len(out_samples))))
					rx_pkts.append((now_m, 1))
					# Emit stats at most once per second to keep UDP overhead low.
					if now_m - last_stat_mono >= float(ccfg.stat_emit_interval_seconds):
						send_stat()
						last_stat_mono = now_m
				except Exception:
					pass
			except Exception as e:
				send({"type": "log", "msg": f"[Collector] decode error: {e!r}"})

	main_loop = asyncio.get_running_loop()
	consumer_task = asyncio.create_task(process_payloads())

	# Simple reconnect loop with backoff
	backoff_s = float(ccfg.initial_backoff_seconds)
	try:
		while True:
			try:
				send({"type": "log", "msg": f"[Collector] Connecting to {device} (mode={mode}, rate={rate})..."})
				# After a drop, BlueZ may need extra time for the peripheral to start advertising again.
				# Also, give discovery a bit more slack than the default.
				await client.connect(device, connection_timeout=float(ccfg.connect_timeout_seconds))
				connected_addr = device
				send({"type": "log", "msg": "[Collector] Connected. Sending HELLO..."})
				await client.hello(ref_id=int(ccfg.hello_ref_id))
				send({"type": "log", "msg": "[Collector] Subscribing IMU..."})
				# Movesense example uses fixed ref; keep stable.
				sub = await client.subscribe_imu(
					sample_rate_hz=rate,
					mode=mode,
					on_data=on_data,
					ref_id=int(ccfg.subscribe_ref_id),
				)
				send({"type": "log", "msg": "[Collector] Subscribed. Streaming."})

				# Watchdog loop: if notifications stop, reconnect.
				while True:
					await asyncio.sleep(1.0)
					stale = time.monotonic() - last_notify_mono
					if stale > float(ccfg.notify_stale_timeout_seconds):
						raise RuntimeError(f"IMU notifications stale for {stale:.1f}s")
			except asyncio.CancelledError:
				raise
			except Exception as e:
				last_error = repr(e)
				send_stat()
				send({"type": "log", "msg": f"[Collector] Stream error: {e!r}. Reconnecting..."})
				# Best effort: unsubscribe + disconnect
				try:
					await client.disconnect()
				except Exception:
					pass
				# Recreate Bleak client on every reconnect to avoid stale D-Bus/BlueZ handles.
				try:
					client = mk_client()
				except Exception:
					pass

				await asyncio.sleep(backoff_s)
				backoff_s = min(float(ccfg.max_backoff_seconds), backoff_s * float(ccfg.backoff_multiplier))
				# If device was specified by name and failed, keep same string (Bleak resolves again).
				if connected_addr is None:
					connected_addr = device
	except asyncio.CancelledError:
		raise
	finally:
		try:
			consumer_task.cancel()
			await consumer_task
		except Exception:
			pass
		try:
			await client.disconnect()
		except Exception:
			pass
		try:
			sock.close()
		except Exception:
			pass


def main(argv: Optional[list[str]] = None) -> int:
	cfg = get_config()
	p = argparse.ArgumentParser(description="Movesense IMU collector (separate process)")
	p.add_argument("--device", required=True, help="Movesense MAC or name")
	p.add_argument("--mode", default=cfg.movesense.default_mode, help="IMU mode: IMU6 or IMU9")
	p.add_argument("--rate", type=int, default=int(cfg.movesense.default_rate), help="Sample rate (Hz)")
	p.add_argument("--udp-host", default=cfg.imu_udp.host, help="UDP host for server")
	p.add_argument("--udp-port", type=int, default=int(cfg.imu_udp.port), help="UDP port for server")
	args = p.parse_args(argv)

	try:
		asyncio.run(_run_collector(args.device, args.mode, int(args.rate), args.udp_host, int(args.udp_port)))
		return 0
	except KeyboardInterrupt:
		return 0
	except Exception as e:
		logger.exception("[Collector] fatal: %s", e)
		return 2


if __name__ == "__main__":
	raise SystemExit(main())


