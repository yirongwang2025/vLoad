from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import time
from collections import deque
from typing import Any, Dict, Optional

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
			cut = now_m - 5.0
			while rx_window and rx_window[0][0] < cut:
				rx_window.popleft()
			while rx_pkts and rx_pkts[0][0] < cut:
				rx_pkts.popleft()
			s5 = sum(n for _, n in rx_window)
			p5 = len(rx_pkts)
			rate5 = float(s5) / 5.0
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
	payload_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=600)
	last_notify_mono = time.monotonic()

	def on_data(payload: bytes) -> None:
		# Keep callback minimal; Bleak may invoke from another thread.
		nonlocal last_notify_mono
		last_notify_mono = time.monotonic()
		# Bleak may invoke this on a non-async thread. We capture the main loop once.
		try:
			main_loop.call_soon_threadsafe(payload_q.put_nowait, payload)
		except Exception:
			# Drop if queue is full or loop is closing.
			pass

	# Per-sample timestamping state
	last_sample_t = 0.0

	async def process_payloads() -> None:
		nonlocal last_sample_t
		last_stat_mono = time.monotonic()
		while True:
			payload = await payload_q.get()
			try:
				decoded = client.decode_imu_payload(payload, mode)  # type: ignore[attr-defined]
				samples = decoded.get("samples", []) or []
				frame_ts = decoded.get("timestamp")
				seq = decoded.get("seq")
				if not samples:
					continue

				now_frame_t = time.time()
				n = len(samples)
				dt = 1.0 / float(rate) if rate and rate > 0 else 1.0 / 104.0

				out_samples = []
				for i, s in enumerate(samples):
					# Distribute timestamps across the packet, end at now_frame_t, enforce monotonicity.
					t_i = now_frame_t - float((n - 1 - i)) * dt
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
					}
				)

				# Update collector rx rate window (after successful decode)
				try:
					now_m = time.monotonic()
					rx_window.append((now_m, int(len(out_samples))))
					rx_pkts.append((now_m, 1))
					# Emit stats at most once per second to keep UDP overhead low.
					if now_m - last_stat_mono >= 1.0:
						send_stat()
						last_stat_mono = now_m
				except Exception:
					pass
			except Exception as e:
				send({"type": "log", "msg": f"[Collector] decode error: {e!r}"})

	main_loop = asyncio.get_running_loop()
	consumer_task = asyncio.create_task(process_payloads())

	# Simple reconnect loop with backoff
	backoff_s = 0.5
	try:
		while True:
			try:
				send({"type": "log", "msg": f"[Collector] Connecting to {device} (mode={mode}, rate={rate})..."})
				# After a drop, BlueZ may need extra time for the peripheral to start advertising again.
				# Also, give discovery a bit more slack than the default.
				await client.connect(device, connection_timeout=25.0)
				connected_addr = device
				send({"type": "log", "msg": "[Collector] Connected. Sending HELLO..."})
				await client.hello(ref_id=1)
				send({"type": "log", "msg": "[Collector] Subscribing IMU..."})
				# Movesense example uses fixed ref; keep stable.
				sub = await client.subscribe_imu(sample_rate_hz=rate, mode=mode, on_data=on_data, ref_id=99)
				send({"type": "log", "msg": "[Collector] Subscribed. Streaming."})

				# Watchdog loop: if notifications stop, reconnect.
				while True:
					await asyncio.sleep(1.0)
					stale = time.monotonic() - last_notify_mono
					if stale > 6.0:
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
				backoff_s = min(8.0, backoff_s * 1.6)
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
	p = argparse.ArgumentParser(description="Movesense IMU collector (separate process)")
	p.add_argument("--device", required=True, help="Movesense MAC or name")
	p.add_argument("--mode", default="IMU9", help="IMU mode: IMU6 or IMU9")
	p.add_argument("--rate", type=int, default=104, help="Sample rate (Hz)")
	p.add_argument("--udp-host", default="127.0.0.1", help="UDP host for server")
	p.add_argument("--udp-port", type=int, default=9999, help="UDP port for server")
	args = p.parse_args(argv)

	try:
		asyncio.run(_run_collector(args.device, args.mode, int(args.rate), args.udp_host, int(args.udp_port)))
		return 0
	except KeyboardInterrupt:
		return 0
	except Exception as e:
		print(f"[Collector] fatal: {e!r}", file=sys.stderr)
		return 2


if __name__ == "__main__":
	raise SystemExit(main())


