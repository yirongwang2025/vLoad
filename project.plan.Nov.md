## vLoad – Movesense IMU Web Monitor & Jump Detection

vLoad is a FastAPI‑based web application for **real‑time monitoring of Movesense IMU sensors** and, in later phases, **figure‑skating jump detection** based on IMU9 data.

The work is organized into explicit phases so we can validate each layer (BLE, decoding, plotting, detection, deployment) independently.

---

## Phase 0 – Reliable Streaming & Web UI (COMPLETED)

**Goals**
- Robust BLE connection to a single Movesense sensor.
- Correct decoding of IMU6 / IMU9 notifications using the official Movesense GATT SensorData Protocol (GSP) layout.
- Simple, responsive web UI that visualizes acceleration, gyro, and magnetometer data.

**Key components**
- `movesense_gatt.py`
  - `MovesenseGATTClient` wraps Bleak and GSP:
    - `scan_for_movesense(...)` – cross‑platform scan for devices whose name contains `"Movesense"`.
    - `connect(address_or_name)` – connects by MAC or by name, verifying GSP service/characteristics.
    - `hello()` – sends a simple HELLO frame.
    - `subscribe_imu(mode, sample_rate_hz, on_data)` – subscribes to IMU6/IMU9 streams.
  - `decode_imu_payload(payload, mode)`:
    - Handles two‑packet GSP frames:
      - `PACKET_TYPE_DATA (2)` + `PACKET_TYPE_DATA_PART2 (3)`.
      - Combines the packets and parses:
        - `[type][ref][timestamp][acc block][gyro block][mag block]`.
        - 8 samples × 3 axes × `float32` for each block (IMU9) or acc+gyro only (IMU6).
    - Falls back to a “legacy” flat‑float decoder for non‑GSP / unknown layouts.
    - Returns:
      - `{"type", "ref", "seq", "timestamp", "samples: [ {"acc": [...], "gyro": [...], "mag": [...]}, ... ] }`.
- `server.py`
  - FastAPI app with:
    - `GET /` – serves the single‑page HTML UI.
    - `GET /scan` – BLE scan, returns `{ devices: [{address, name}, ...] }`.
    - `POST /connect` – starts a long‑running BLE worker (async task) for the requested device.
    - `POST /disconnect` – stops the worker and disconnects the client.
    - `WebSocket /ws` – pushes IMU samples and log messages to the browser.
  - `ble_worker(device, mode, rate)`:
    - Connects explicitly by MAC/name; if that fails, falls back to auto‑scan.
    - Sends `HELLO`, then subscribes to `Meas/IMU9/104` (or IMU6) via `subscribe_imu`.
    - `on_data` callback:
      - Uses `client.decode_imu_payload(...)`.
      - Builds a compact JSON with:
        - `mode`, `rate`, `timestamp`, `seq`, `samples_len`.
        - `first_sample` (acc/gyro/mag of the first sample in the frame).
        - A very small “analysis” dict (e.g., acceleration magnitude) for demo.
      - Broadcasts the JSON to all connected WebSocket clients.
    - Runs in a `while True: sleep(3600)` loop until cancelled or disconnected.
  - `INDEX_HTML` (inline HTML/JS):
    - Device control:
      - Text input for MAC / name.
      - `Scan`, `Connect`, `Disconnect` buttons.
      - Last‑used device persisted in `localStorage`.
    - Live plots:
      - Three `<canvas>` elements (Accel, Gyro, Mag; X/Y/Z for each).
      - Simple custom plotting with dynamic Y‑scaling and legend.
      - History window of ~150 points, redraw throttled to ~30 FPS.
    - Log window:
      - `<pre id="logBox">` that accumulates log lines from the server and UI.
      - All BLE state changes and errors are visible without opening the browser devtools.

**Outcome**
- Web UI can **scan, connect, stream, and visualize** IMU9/IMU6 data at 104 Hz.
- BLE errors and connection state are clearly visible in the log area.

---

## Phase 1 – Vertical Tracking Diagnostics (COMPLETED)

**Goals**
- Add a **lightweight, non‑intrusive real‑time layer** that tracks vertical acceleration and vertical gyro.
- Confirm that the Z‑axis and gravity compensation behave as expected, without yet attempting jump detection.

**Key components**
- `modules/jump_detector.py`
  - `JumpDetectorRealtime`:
    - Maintains a rolling window (`deque`) of:
      - `{"t": wall_clock_time, "az": acc_z, "gz": gyro_z}`.
    - `update(sample)` expects:
      - `{"t": <float>, "acc": [ax, ay, az], "gyro": [gx, gy, gz], "mag": [...] (optional)}`.
    - After collecting ≈1 second of samples:
      - Computes:
        - `max|az|`
        - `max|az − g|`
        - `max|gz|`
      - Emits a **single log message** via its `logger` callback:
        - `"[Phase1] Vertical tracking active over ~Xs: max|az|=..., max|az-g|=..., max|gz|=..."`.
      - Sets an internal flag so it **only logs once per connection**.
    - Always returns an empty list (`[]`) – jump events will be added in later phases.
- Integration in `server.py`
  - In `ble_worker`:
    - Create one `JumpDetectorRealtime` per worker:
      - `jump_detector = JumpDetectorRealtime(sample_rate_hz=rate, window_seconds=3.0, logger=lambda msg: _log_to_clients(f"[JumpDetector] {msg}"))`.
    - In the decoded `on_data`:
      - For each IMU sample in the notification, call `jump_detector.update(...)`.
      - Any exceptions inside the detector are caught so they **cannot break the BLE stream**.
  - The WebSocket JSON payload is **unchanged**, so plots and the rest of the UI behave exactly as in Phase 0.

**Outcome**
- You see a one‑time log line like:

  > `[JumpDetector] [Phase1] Vertical tracking active over ~0.95 s: max|az|=..., max|az-g|=..., max|gz|=...`

  confirming that:
  - Z acceleration and Z gyro are being tracked correctly.
  - Gravity‑adjusted vertical acceleration (`az − 9.8`) behaves as expected.

---

## Phase 2 – IMU‑Only Jump Detection (PLANNED, BRUENING‑ALIGNED)

**Goals**
- Implement a **clean, IMU‑only jump detection algorithm** suitable for real‑time use, explicitly aligned with:
  - *Bruening et al., 2018 – “A sport‑specific wearable jump monitor for figure skating”* (`Bruening2018.pdf` in this repo).
- Use only IMU9 streams (acc + gyro + mag), no video.

**Planned steps (mirroring Bruening et al.)**

1. **Signal preprocessing (Step 2.1 – COMPLETED, `modules/web_jump_detection.py`)**
   - Work primarily in the **vertical direction**:
     - \(a_z\) = vertical acceleration from the sensor’s Z‑axis (approx. aligned to body vertical).
     - \(a_z - g\) = gravity‑removed vertical acceleration.
     - \(\omega_z\) = gyro about the vertical axis.
   - Currently implemented:
     - `preprocess_vertical_series(buffer)` takes the rolling buffer from `JumpDetectorRealtime` and produces:
       - Series: `t`, `az`, `az_no_g`, `gz`.
       - Summary stats: `count`, `duration`, `max_abs_az`, `max_abs_az_no_g`, `max_abs_gz`.
     - Phase 1 and Phase 2.1 log lines in `JumpDetectorRealtime` confirm sensible ranges once per connection.
   - Future addition in this step family:
     - Apply smoothing filters (e.g. Savitzky–Golay / Butterworth) tuned to jump time‑scales (~3–10 Hz) so take‑off and landing peaks are clear.

2. **Vertical peak counting diagnostics (Step 2.2 – COMPLETED, `modules/web_jump_detection.py` + `modules/jump_detector.py`)**
   - Implemented a simple, SciPy‑free peak finder:
     - `find_vertical_peaks(preprocessed, min_height, min_distance_samples)` works on \(|a_z - g|\) and enforces:
       - Minimum peak height (`min_height`, in m/s²).
       - Minimum spacing (`min_distance_samples`, derived from `min_peak_distance_s` and sample rate).
   - `JumpDetectorRealtime`:
     - Maintains parameters:
       - `min_peak_height_m_s2` (default 3.0 m/s²).
       - `min_peak_distance_s` (default 0.18 s).
     - About once per second, logs a diagnostic line such as:
       - `"[Phase2.2] Peaks in ~T s window: K (min_height=..., min_spacing=...)"`,
       confirming that clear jumps produce more peaks than quiet skating/standing.

3. **Jump‑candidate identification from vertical acceleration (Step 2.3 – COMPLETED)**
   - Use the vertical peaks found in Step 2.2 to form **peak pairs**:
     - Treat earlier peaks as candidate “take‑off” events and later peaks as candidate “landing” events.
     - Accept a pair if the time between peaks lies within a plausible **flight‑time window** (e.g. 0.25–0.8 s, tunable).
   - Implemented in:
     - `modules/web_jump_detection.py`:
       - `build_jump_windows(preprocessed, peak_indices, min_flight_time_s, max_flight_time_s)` forms windows with:
         - `(i_takeoff, i_landing, t_takeoff, t_landing, flight_time)`.
     - `modules/jump_detector.py` (`JumpDetectorRealtime`):
       - Every ~2 seconds logs a `[Phase2.3]` summary like:
         - `"[Phase2.3] Candidate windows in ~T s: M (T_f: ...)"`,
         confirming that clear jump sequences produce a small number of candidate windows with realistic flight times while quiet motion produces few or none.
   - Later steps use these windows as the basis for computing jump height and rotation metrics and, eventually, final detected jumps.

4. **Flight time and jump height, from projectile motion (Step 2.4 – COMPLETED)**
   - For each accepted jump candidate:
     - Flight time \(T_f\) = time between the two vertical‑acc peaks.
     - Jump height via the same physics used in the paper, e.g.
       \[
       h = \frac{g T_f^2}{8}
       \]
       (or a closely related peak‑to‑peak scaling), assuming roughly symmetric take‑off/landing.
   - Store and report **both** flight time and derived height, emphasizing that **relative changes** across jumps/sessions are more reliable than absolute height.
   - Implemented in:
     - `modules/web_jump_detection.py`:
       - `compute_window_metrics(preprocessed, windows)` enriches each candidate with:
         - `height`, `peak_az_no_g` (max |a_z-g|), `peak_gz` (max |ω_z|), and `t_peak_gz`.
     - `modules/jump_detector.py`:
       - Phase 2.4 log line summarizing the first few candidates, e.g.:
         - `"[Phase2.4] Jump metrics: T_f=..., h=..., ωz=...°/s | ..."` for quick sanity‑checking of magnitude and timing.

5. **Rotation speed and timing within the jump (Step 2.5 – COMPLETED within metrics above)**
   - Within the jump flight window (between the vertical‑acc peaks):
     - Compute \( \omega_z(t) \) from gyro; find **peak rotation speed** = max \(|\omega_z|\).
     - Record the **phase** of this peak within flight:
       - `rotation_phase = (t_peak − t_takeoff) / T_f` (expected to be near ~0.6–0.7 in many jumps, as reported in the paper).
   - Use these metrics for both performance feedback (rotation quality) and as an extra check that the event is a genuine multi‑revolution jump.

6. **False‑positive control and confidence scoring (Step 2.6 – COMPLETED)**
   - Goals:
     - Reject candidate windows that are too small (height/impulse/rotation) or too close together in time.
     - Assign a simple confidence score based on height, vertical impulse, and rotation speed.
   - Implementation plan:
     - `modules/web_jump_detection.py`:
       - `select_jump_events(windows_with_metrics, ...)`:
         - Filters on:
           - Minimum height.
           - Minimum `peak_az_no_g`.
           - Minimum `peak_gz` (rotation speed).
         - Enforces a minimum separation between emitted jumps.
         - Computes a heuristic confidence in \([0, 1]\) from normalized height, vertical impulse, and rotation.
     - `modules/jump_detector.py`:
       - Extend `JumpDetectorRealtime.update(...)` to:
         - Call `select_jump_events(...)` on the metrics from Step 2.4.
         - Keep track of the last emitted jump time to avoid duplicates across overlapping buffers.
         - Log concise `[Jump]` lines with:
           - Peak time, flight time, height, peak rotation speed, rotation phase, and confidence.
         - Return a list of new jump events so the server/UI can consume them later.

7. **Real‑time integration (Step 2.7 - COMPLETED)**
   - Once Steps 2.1–2.5 are validated, wire jump events into the FastAPI/WebSocket path:
     - Forward jump events from `JumpDetectorRealtime` to the browser as dedicated messages.
     - Add a simple jump list or counter in the UI, using the existing log as the primary debug channel.

**Output format (tentative)**
- Each detected jump will be a small dict, e.g.:

  ```json
  {
    "type": "jump",
    "t_peak": 1733000000.123,
    "flight_time": 0.45,
    "height": 0.22,
    "acc_peak": 18.5,
    "gyro_peak": 900.0,
    "rotation_phase": 0.64,
    "confidence": 0.87
  }
  ```

These will initially be **logged for validation** (and cross‑checked with manual video/experience), then later surfaced in dedicated jump‑analysis views in the web UI.

---

## Phase 3 – UI & Deployment Enhancements (PLANNED)

**Goals**
- Polish the web UI for day‑to‑day use and set up reliable production deployment.

**Planned items**
- UI:
  - Dedicated jump timeline / table in the browser.
  - Per‑jump details panel (acc/gyro plots around the event, confidence, rotation estimate).
  - Export of IMU and detected jumps to CSV or JSON.
- Deployment:
  - `deploy.sh` script that automates:
    - Pulling latest code from GitHub.
    - Restarting the `systemd` service running `uvicorn`.
  - `systemd` unit file to keep the server running in the background.
  - `nginx` config to terminate TLS and reverse‑proxy WebSocket traffic to uvicorn.

---

## Running the App Locally

Assuming you have Python 3.13 and a Movesense sensor:

```bash
cd ~/vLoad
source venv/bin/activate        # or .\venv\Scripts\activate on Windows

python -m uvicorn server:app --host 0.0.0.0 --port 8080
```

Then open the browser to `http://<server-ip>:8080`, click **Scan**, select or confirm the MAC, and click **Connect**.  
You should see the three plots updating and, after a second or so, a `[JumpDetector][Phase1]` line in the log.

