import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Iterable, Tuple, Dict, Any

from bleak import BleakScanner, BleakClient, BleakError

import struct

# Movesense GATT SensorData Protocol (GSP)
# Service and characteristic UUIDs (public, stable)
GSP_SERVICE_UUID = "34802252-7185-4d5d-b431-630e7050e8f0"
GSP_RX_CHAR_UUID = "34800001-7185-4d5d-b431-630e7050e8f0"  # Client → Sensor (Write)
GSP_TX_CHAR_UUID = "34800002-7185-4d5d-b431-630e7050e8f0"  # Sensor → Client (Notify)


# Protocol opcodes (from public documentation)
OPCODE_HELLO = 0x00
OPCODE_SUBSCRIBE = 0x01
OPCODE_UNSUBSCRIBE = 0x02


def _default_logger(msg: str) -> None:
	"""
	Simple default logger that writes to stdout.
	"""
	print(msg)


@dataclass
class Subscription:
	"""
	Represents a GSP subscription.
	"""
	ref_id: int
	resource_path: str


class MovesenseGATTClient:
	"""
	Python BLE client for the Movesense GATT SensorData Protocol (GSP).

	Notes:
	- Requires Bluetooth LE and the 'bleak' library.
	- IMU data is reported via notifications on GSP_TX_CHAR_UUID.
	- Payloads for measurement streams may be in SBEM (binary) format; this client
	  exposes raw bytes to your callback. You can parse them as needed.

	Basic usage:
	    client = MovesenseGATTClient()
	    await client.connect(address_or_name="Movesense 123456")
	    await client.hello()
	    sub = await client.subscribe_imu(mode="IMU6", sample_rate_hz=104, on_data=my_cb)
	    # ... receive data ...
	    await client.unsubscribe(sub)
	    await client.disconnect()
	"""

	def __init__(self, logger: Callable[[str], None] = _default_logger):
		self._client: Optional[BleakClient] = None
		self._log = logger
		self._notify_handler_set = False
		self._user_data_callback: Optional[Callable[[bytes], None]] = None
		self._next_ref_id = 1
		# For multi-part GSP frames (e.g., IMU9 uses DATA + DATA_PART2)
		self._pending_frames: Dict[int, bytes] = {}
		# Optional callback invoked when Bleak signals the device disconnected.
		# Signature varies by backend; we normalize to a no-arg callable for callers.
		self._on_disconnected: Optional[Callable[[], None]] = None

	def set_disconnected_callback(self, cb: Optional[Callable[[], None]]) -> None:
		"""
		Set a callback to be invoked when the BLE link disconnects.
		"""
		self._on_disconnected = cb

	@property
	def is_connected(self) -> bool:
		return bool(self._client and self._client.is_connected)

	# -------------------------------
	# Discovery / connection helpers
	# -------------------------------
	@staticmethod
	async def scan_for_movesense(timeout: float = 5.0) -> Iterable[Tuple[str, str]]:
		"""
		Scan for nearby Movesense devices and yield (address, name).
		Compatible across Bleak versions and backends.
		"""
		found_any = False
		
		# Try Bleak's return_adv mode (varies by version/OS)
		try:
			logging.debug(f"Scanning with return_adv=True, timeout={timeout}s...")
			results = await BleakScanner.discover(timeout=timeout, return_adv=True)  # Bleak >= 0.22
			# results may be: dict[BLEDevice, AdvertisementData], list[(BLEDevice, AdvertisementData)],
			# or list[(BLEDevice, AdvertisementData, rssi)]
			iterable = None
			if isinstance(results, dict):
				iterable = results.items()
				logging.debug(f"Scan returned dict with {len(results)} device(s)")
			else:
				iterable = results  # assume iterable of tuples or devices
				logging.debug(f"Scan returned list/iterable")
			
			device_count = 0
			for item in iterable:
				device_count += 1
				device = None
				adv = None
				# Normalize item shape
				if isinstance(item, tuple):
					if len(item) >= 2:
						device, adv = item[0], item[1]
					elif len(item) == 1:
						device = item[0]
				else:
					device = item
				name = ""
				if adv is not None:
					name = getattr(adv, "local_name", None) or ""
				if not name and device is not None:
					name = getattr(device, "name", "") or ""
				
				logging.debug(f"Found device: {device.address if device else 'unknown'} - '{name}'")
				
				service_uuids = []
				if adv is not None:
					try:
						service_uuids = [str(u).lower() for u in (getattr(adv, 'service_uuids', None) or [])]
					except Exception:
						service_uuids = []
				
				is_movesense = False
				if name and 'movesense' in name.lower():
					is_movesense = True
				elif service_uuids and GSP_SERVICE_UUID.lower() in service_uuids:
					is_movesense = True
				
				if is_movesense:
					found_any = True
					logging.debug(f"  -> Match! Yielding Movesense device")
					if not name and device is not None:
						name = f"Movesense ({device.address})"
					yield device.address, name
			logging.debug(f"Scanned {device_count} total device(s), found {1 if found_any else 0} Movesense")
			
			# If we successfully used return_adv mode and found devices, we're done
			if found_any:
				return
			# If return_adv worked but found nothing, fall through to try the basic method
			logging.debug("No Movesense found with return_adv, trying fallback method...")
			
		except TypeError:
			# Older Bleak without return_adv parameter
			logging.debug("return_adv not supported, using fallback method")
			pass
		except Exception as e:
			# Be resilient; fall back to basic discovery
			logging.debug(f"return_adv scan failed: {e}, trying fallback")
			pass

		# Fallback for older Bleak: use device.name and (optionally) legacy metadata if present
		logging.debug(f"Scanning with basic discover, timeout={timeout}s...")
		devices = await BleakScanner.discover(timeout=timeout)
		logging.debug(f"Basic scan found {len(devices)} device(s)")
		for d in devices:
			name = getattr(d, "name", "") or ""
			if not name:
				md = getattr(d, "metadata", None)
				if isinstance(md, dict):
					name = md.get("local_name", "") or ""
			logging.debug(f"Found device: {d.address} - '{name}'")
			if "movesense" in name.lower():
				logging.debug(f"  -> Match! Yielding Movesense device")
				yield d.address, (name or d.address)

	async def connect(self, address_or_name: str, connection_timeout: float = 20.0) -> None:
		"""
		Connect to a Movesense device by MAC address or by name.
		For MAC addresses, perform discovery first so BlueZ can resolve the device.
		"""
		# Normalize input (strip accidental whitespace/newlines from env or args)
		address_or_name = (address_or_name or "").strip()
		target_address = address_or_name
		ble_device = None

		if ":" in address_or_name:
			# Resolve MAC by scanning to populate BlueZ cache
			self._log(f"Attempting MAC-based connect to '{address_or_name}'...")
			try:
				ble_device = await BleakScanner.find_device_by_address(address_or_name, timeout=10.0)
			except Exception:
				ble_device = None
			if ble_device is None:
				self._log("Direct find_device_by_address failed; falling back to discovery scan.")
				devices = await BleakScanner.discover(timeout=7.0)
				for d in devices:
					if d.address.replace("-", ":").lower() == address_or_name.lower():
						self._log(f"Matched address via discovery: {d.address}")
						ble_device = d
						target_address = d.address
						break
				if ble_device is None:
					# As a final fallback, scan for any Movesense device and connect to it.
					self._log("MAC not found; scanning for any Movesense device as fallback.")
					found = [item async for item in self.scan_for_movesense(timeout=7.0)]
					if found:
						addr, name = found[0]
						self._log(f"Found Movesense '{name}' at {addr}; connecting to it.")
						target_address = addr
						try:
							ble_device = await BleakScanner.find_device_by_address(addr, timeout=5.0)
						except Exception:
							ble_device = None
					if ble_device is None:
						raise BleakError(f"Device with address {address_or_name} was not found. Ensure it is advertising and nearby.")
		else:
			self._log(f"Scanning for device named '{address_or_name}'...")
			async for addr, name in self.scan_for_movesense(timeout=7.0):
				if name.strip().lower() == address_or_name.strip().lower():
					target_address = addr
					try:
						ble_device = await BleakScanner.find_device_by_address(addr, timeout=5.0)
					except Exception:
						ble_device = None
					break
			if ble_device is None:
				raise BleakError(f"Could not find Movesense named '{address_or_name}'.")

		self._log(f"Connecting to Movesense at {target_address} ...")

		# Bleak supports a disconnected_callback on most backends, but the constructor
		# signature differs across versions. Be defensive.
		def _disconnected_cb(*_: Any, **__: Any) -> None:
			try:
				self._log("BLE disconnected callback fired.")
			except Exception:
				pass
			try:
				if self._on_disconnected:
					self._on_disconnected()
			except Exception:
				pass

		try:
			self._client = BleakClient(ble_device or target_address, timeout=connection_timeout, disconnected_callback=_disconnected_cb)
		except TypeError:
			# Older Bleak versions may not accept disconnected_callback in ctor.
			self._client = BleakClient(ble_device or target_address, timeout=connection_timeout)
		await self._client.__aenter__()

		if not self._client.is_connected:
			raise BleakError(f"Failed to connect to {target_address}")

		# Verify the GSP service exists (handle multiple Bleak versions)
		services = None
		try:
			services = await self._client.get_services()  # new-ish Bleak
		except AttributeError:
			services = getattr(self._client, "services", None)  # older Bleak exposes .services

		uuid_set = set()
		if services is not None:
			try:
				# BleakGATTServiceCollection (common): iterate underlying dict
				for s in getattr(services, "services", {}).values():
					uuid_set.add((getattr(s, "uuid", "") or "").lower())
			except Exception:
				try:
					# Iterable of services
					for s in services:
						uuid_set.add((getattr(s, "uuid", "") or "").lower())
				except Exception:
					pass

		# Only enforce if we actually gathered UUIDs
		if uuid_set:
			if GSP_SERVICE_UUID.lower() not in uuid_set and all(u.lower() not in uuid_set for u in [GSP_RX_CHAR_UUID, GSP_TX_CHAR_UUID]):
				await self.disconnect()
				raise BleakError("GSP service/characteristics not found on device.")

		self._log("Connected.")

	async def disconnect(self) -> None:
		"""
		Disconnect cleanly.
		"""
		if self._client is not None:
			try:
				if self._notify_handler_set:
					try:
						await self._client.stop_notify(GSP_TX_CHAR_UUID)
					except Exception:
						pass
				try:
					await self._client.__aexit__(None, None, None)
				except Exception:
					# Ignore D-Bus EOFs or backend teardown races
					pass
			finally:
				self._client = None
				self._notify_handler_set = False
				self._user_data_callback = None
				self._log("Disconnected.")

	# -------------------------------
	# Protocol helpers
	# -------------------------------
	def _allocate_ref(self) -> int:
		ref_id = self._next_ref_id
		self._next_ref_id = (self._next_ref_id + 1) & 0xFF
		if self._next_ref_id == 0:
			self._next_ref_id = 1  # keep non-zero
		return ref_id

	async def _ensure_notifications(self, on_data: Optional[Callable[[bytes], None]]) -> None:
		if not self._client:
			raise BleakError("Not connected.")

		if not self._notify_handler_set:
			def _handler(_: int, data: bytearray) -> None:
				# Pass raw data to the user callback. Higher-level parsing of GSP/SBEM can be added here.
				if self._user_data_callback:
					try:
						self._user_data_callback(bytes(data))
					except Exception as e:
						self._log(f"Data callback error: {e}")

			await self._client.start_notify(GSP_TX_CHAR_UUID, _handler)
			self._notify_handler_set = True

		if on_data:
			self._user_data_callback = on_data

	async def hello(self, ref_id: Optional[int] = None) -> None:
		"""
		Send HELLO to verify link and fetch basic info.
		"""
		if not self._client:
			raise BleakError("Not connected.")

		ref = (ref_id if ref_id is not None else self._allocate_ref()) & 0xFF
		frame = self._encode_hello(ref)
		await self._client.write_gatt_char(GSP_RX_CHAR_UUID, frame, response=True)
		self._log(f"HELLO sent (ref={ref}).")

	async def subscribe(self, resource_path: str, on_data: Callable[[bytes], None], ref_id: Optional[int] = None) -> Subscription:
		"""
		Subscribe to a Movesense measurement resource, e.g., 'Meas/IMU6/104'.

		Note: resource_path MUST NOT start with a leading slash for GSP (use 'Meas/IMU6/104').
		"""
		if not self._client:
			raise BleakError("Not connected.")

		await self._ensure_notifications(on_data)

		ref = (ref_id if ref_id is not None else self._allocate_ref()) & 0xFF
		frame = self._encode_subscribe(ref, resource_path)
		await self._client.write_gatt_char(GSP_RX_CHAR_UUID, frame, response=True)
		self._log(f"SUBSCRIBE sent for '{resource_path}' (ref={ref}).")
		return Subscription(ref_id=ref, resource_path=resource_path)

	async def unsubscribe(self, subscription: Subscription) -> None:
		"""
		Unsubscribe a previous subscription.
		"""
		if not self._client:
			raise BleakError("Not connected.")
		frame = self._encode_unsubscribe(subscription.ref_id)
		await self._client.write_gatt_char(GSP_RX_CHAR_UUID, frame, response=True)
		self._log(f"UNSUBSCRIBE sent (ref={subscription.ref_id}).")

	# -------------------------------
	# IMU convenience methods
	# -------------------------------
	async def subscribe_imu(
		self,
		sample_rate_hz: int = 104,
		mode: str = "IMU6",
		on_data: Optional[Callable[[bytes], None]] = None,
		ref_id: Optional[int] = None,
	) -> Subscription:
		"""
		Convenience wrapper to subscribe to IMU streams.
		- mode: 'IMU6' (acc+gyro) or 'IMU9' (acc+gyro+mag)
		- sample_rate_hz: commonly one of 13, 26, 52, 104, 208

		Notes from Movesense official examples:
		- They use a *fixed* ref_id (e.g. 99 for IMU9) and unsubscribe with the same ref.
		  Using a fixed ref_id reduces the risk of accidentally having multiple active
		  subscriptions if unsubscribe fails due to a transient disconnect.
		"""
		mode_norm = mode.strip().upper()
		if mode_norm not in ("IMU6", "IMU9"):
			raise ValueError("mode must be 'IMU6' or 'IMU9'")
		# Movesense official examples send a leading slash in the URI.
		resource = f"/Meas/{mode_norm}/{sample_rate_hz}"
		return await self.subscribe(
			resource_path=resource,
			on_data=(on_data or self._default_imu_print),
			ref_id=ref_id,
		)

	def _default_imu_print(self, payload: bytes) -> None:
		"""
		Default handler: print raw hex payload for debugging.
		Consider replacing with an SBEM decoder for structured IMU data.
		"""
		self._log(f"IMU data ({len(payload)} bytes): {payload.hex()[:256]}{'...' if len(payload) > 128 else ''}")

	# -------------------------------
	# Frame encoding (conservative)
	# -------------------------------
	@staticmethod
	def _encode_hello(ref: int) -> bytes:
		"""
		Encode a HELLO frame.
		Frame layout (conservative, per public docs):
		    [OPCODE_HELLO:1][REF:1]
		"""
		return bytes([OPCODE_HELLO & 0xFF, ref & 0xFF])

	@staticmethod
	def _encode_subscribe(ref: int, resource_path: str) -> bytes:
		"""
		Encode a SUBSCRIBE frame for a resource path, e.g., 'Meas/IMU6/104'.
		Conservative framing (keeps compatibility with known public examples):
		    [OPCODE_SUBSCRIBE:1][REF:1][URI:ascii...]
		"""
		# Keep leading slash if present; Movesense official examples include it.
		uri_bytes = resource_path.encode("utf-8")
		return bytes([OPCODE_SUBSCRIBE & 0xFF, ref & 0xFF]) + uri_bytes

	@staticmethod
	def _encode_unsubscribe(ref: int) -> bytes:
		"""
		Encode an UNSUBSCRIBE frame.
		    [OPCODE_UNSUBSCRIBE:1][REF:1]
		"""
		return bytes([OPCODE_UNSUBSCRIBE & 0xFF, ref & 0xFF])

	def decode_imu_payload(self, payload: bytes, mode: str) -> dict:
		"""
		Decode IMU payloads using the official Movesense GATT SensorData Protocol layout.

		The manufacturer Python example (`gatt_sensordata_app`) shows that IMU9 data is
		sent in two packets:
		  - PACKET_TYPE_DATA (2)        → first part, stored
		  - PACKET_TYPE_DATA_PART2 (3)  → second part, combined with the first

		The combined frame then has the following layout (little endian):
		  [type:1][ref:1][timestamp:4][data...]

		For IMU9, `data` contains three contiguous arrays of 8 XYZ float triplets:
		  - 8x acc  (X,Y,Z)
		  - 8x gyro (X,Y,Z)
		  - 8x mag  (X,Y,Z)

		We mirror that logic here and fall back to a legacy best‑effort decoder
		for other packet types or unknown layouts.
		"""
		mode_norm = (mode or "").strip().upper()
		if mode_norm not in ("IMU6", "IMU9"):
			raise ValueError("mode must be 'IMU6' or 'IMU9'")

		PACKET_TYPE_DATA = 0x02
		PACKET_TYPE_DATA_PART2 = 0x03

		if not payload:
			return {"type": None, "ref": None, "seq": None, "timestamp": None, "samples": []}

		packet_type = payload[0]
		ref = payload[1] if len(payload) > 1 else None

		# Multi-packet handling, per Movesense example:
		#   - First part: PACKET_TYPE_DATA
		#   - Second part: PACKET_TYPE_DATA_PART2 (skip its 2‑byte header when joining)
		if packet_type == PACKET_TYPE_DATA:
			# Store first part; wait for DATA_PART2
			if ref is not None:
				self._pending_frames[ref] = payload
			return {"type": packet_type, "ref": ref, "seq": None, "timestamp": None, "samples": []}

		if packet_type == PACKET_TYPE_DATA_PART2:
			if ref is None:
				# Cannot match without ref; drop
				return {"type": packet_type, "ref": None, "seq": None, "timestamp": None, "samples": []}

			first_part = self._pending_frames.pop(ref, None)
			if first_part is None:
				# We never saw the initial DATA packet; mimic manufacturer example by skipping
				self._log(f"Warning: Received DATA_PART2 for ref {ref} without prior DATA. Skipping.")
				return {"type": packet_type, "ref": ref, "seq": None, "timestamp": None, "samples": []}

			# Combine first DATA packet and second DATA_PART2, skipping the second header.
			full_frame = first_part + payload[2:]
			return self._decode_gsp_imu_frame(full_frame, mode_norm)

		# Any other packet types (or single-frame layouts) are interpreted using the old
		# heuristic decoder so that we remain compatible with non‑GSP/legacy firmwares.
		return self._decode_legacy_imu_payload(payload, mode_norm)

	def _decode_gsp_imu_frame(self, frame: bytes, mode_norm: str) -> dict:
		"""
		Decode a complete GSP IMU frame as produced by the manufacturer example.

		Layout (little endian):
		  [type:1][ref:1][timestamp:4][data...]

		For IMU9 (`mode_norm == "IMU9"`):
		  - data length is expected to be ~288 bytes
		  - three blocks of 8 * 3 * 4 bytes:
		      acc block  = 8 samples of XYZ
		      gyro block = 8 samples of XYZ
		      mag block  = 8 samples of XYZ

		For IMU6, we expect two such blocks (acc + gyro) and omit mag.
		"""
		if len(frame) < 6:
			return {"type": None, "ref": None, "seq": None, "timestamp": None, "samples": []}

		packet_type = frame[0]
		ref = frame[1]
		timestamp = int.from_bytes(frame[2:6], "little", signed=False)

		data = frame[6:]
		if not data:
			return {"type": packet_type, "ref": ref, "seq": None, "timestamp": timestamp, "samples": []}

		# One block = 8 samples * 3 axes * 4 bytes
		block_bytes = 8 * 3 * 4  # 96
		data_len = len(data)

		use_mag = False
		if mode_norm == "IMU9" and data_len >= block_bytes * 3:
			use_mag = True
		elif mode_norm == "IMU6" and data_len >= block_bytes * 2:
			use_mag = False
		else:
			# Length does not match the expected pattern; fall back
			return self._decode_legacy_imu_payload(frame, mode_norm)

		samples = []
		sample_count = 8
		gyro_offset = block_bytes
		mag_offset = block_bytes * 2 if use_mag else None

		for i in range(sample_count):
			# Per manufacturer example, each "row" starts at:
			#   offset = 6 + i * 3 * 4
			base = 6 + i * 3 * 4

			try:
				ax, ay, az = struct.unpack_from("<fff", frame, base)
				gx, gy, gz = struct.unpack_from("<fff", frame, base + gyro_offset)
				mag = None
				if use_mag and mag_offset is not None:
					mx, my, mz = struct.unpack_from("<fff", frame, base + mag_offset)
					mag = [float(mx), float(my), float(mz)]
			except struct.error:
				# Truncated frame; stop decoding further samples
				break

			item = {
				"acc": [float(ax), float(ay), float(az)],
				"gyro": [float(gx), float(gy), float(gz)],
			}
			if mag is not None:
				item["mag"] = mag
			samples.append(item)

		return {"type": packet_type, "ref": ref, "seq": None, "timestamp": timestamp, "samples": samples}

	def _decode_legacy_imu_payload(self, payload: bytes, mode_norm: str) -> dict:
		"""
		Best‑effort decoder for legacy/unknown layouts.

		This retains the previous implementation which assumed a flat array of
		float32 samples with an optional 8‑byte header:
		  [type:1][ref:1][seq:2][timestamp:4][floats...]
		"""
		floats_per_sample = 6 if mode_norm == "IMU6" else 9
		bytes_per_sample = floats_per_sample * 4

		p_len = len(payload)
		# Detect 8-byte header [type:1][ref:1][seq:2][timestamp:4]
		if p_len >= 8 and (p_len - 8) % bytes_per_sample == 0:
			header_len = 8
		elif p_len % bytes_per_sample == 0:
			header_len = 0
		elif p_len >= 2 and (p_len - 2) % bytes_per_sample == 0:
			header_len = 2  # rare
		else:
			header_len = 0  # best effort

		info_type = payload[0] if p_len >= 1 else None
		info_ref = payload[1] if p_len >= 2 else None
		info_seq = int.from_bytes(payload[2:4], "little") if header_len >= 4 else None
		info_ts = int.from_bytes(payload[4:8], "little") if header_len >= 8 else None

		data = payload[header_len:]
		num_floats = len(data) // 4
		floats = [f[0] for f in struct.iter_unpack("<f", data[: num_floats * 4])]

		samples = []
		for i in range(0, len(floats), floats_per_sample):
			chunk = floats[i : i + floats_per_sample]
			if len(chunk) < floats_per_sample:
				break
			acc = chunk[0:3]
			gyro = chunk[3:6] if floats_per_sample >= 6 else []
			mag = chunk[6:9] if floats_per_sample == 9 else None
			item = {"acc": acc, "gyro": gyro}
			if mag is not None:
				item["mag"] = mag
			samples.append(item)

		return {"type": info_type, "ref": info_ref, "seq": info_seq, "timestamp": info_ts, "samples": samples}


# ----------------------------------------
# Example CLI usage (python -m vload_modules.movesense_gatt ...)
# ----------------------------------------
async def _demo(address_or_name: Optional[str], rate: int, mode: str, duration_sec: float, do_parse: bool) -> None:
	client = MovesenseGATTClient()
	try:
		if address_or_name:
			await client.connect(address_or_name)
		else:
			# Auto-pick first Movesense found
			print("Scanning for Movesense devices (7 seconds)...")
			found = [item async for item in client.scan_for_movesense(timeout=7.0)]
			print(f"Scan complete. Found {len(found)} Movesense device(s).")
			if not found:
				raise RuntimeError("No Movesense devices found. Ensure the sensor is advertising and nearby.")
			target = found[0]
			print(f"Auto-selecting device: {target[1]} ({target[0]})")
			await client.connect(target[0])

		await client.hello()

		if do_parse:
			def _on_data(payload: bytes) -> None:
				out = client.decode_imu_payload(payload, mode)
				n = len(out["samples"])
				if n:
					s0 = out["samples"][0]
					acc = tuple(round(x, 3) for x in s0["acc"])
					gyro = tuple(round(x, 3) for x in s0["gyro"])
					mag = tuple(round(x, 3) for x in s0.get("mag", [])) if "mag" in s0 else None
					base = f"[PARSED] {n} samples; acc={acc} gyro={gyro}"
					base += f" mag={mag}" if mag is not None else ""
					base += f" ts={out['timestamp']} seq={out['seq']}"
					print(base)
		else:
			def _on_data(payload: bytes) -> None:
				# For demonstration, print a short hex sample and length
				print(f"[DATA] {len(payload)} bytes: {payload.hex()[:64]}{'...' if len(payload) > 32 else ''}")

		sub = await client.subscribe_imu(sample_rate_hz=rate, mode=mode, on_data=_on_data)
		await asyncio.sleep(duration_sec)
		await client.unsubscribe(sub)
	finally:
		await client.disconnect()


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description="Movesense GATT IMU client (BLE, GSP).")
	parser.add_argument("--device", help="Movesense BLE address or exact name (if omitted, auto-select first).")
	parser.add_argument("--rate", type=int, default=104, help="IMU sample rate (e.g., 13, 26, 52, 104, 208).")
	parser.add_argument("--mode", choices=["IMU6", "IMU9"], default="IMU6", help="IMU mode.")
	parser.add_argument("--seconds", type=float, default=10.0, help="Duration to stream before stopping.")
	parser.add_argument("--parse", action="store_true", help="Parse IMU frames and print decoded values.")
	parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
	args = parser.parse_args()
	
	# Configure logging
	if args.debug:
		logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
	else:
		logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

	try:
		asyncio.run(_demo(args.device, args.rate, args.mode, args.seconds, args.parse))
	except KeyboardInterrupt:
		print("\nInterrupted by user.")
	except Exception as e:
		logging.exception("Fatal error: %s", e)
		sys.exit(1)


if __name__ == "__main__":
	main()


