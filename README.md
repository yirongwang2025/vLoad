# vLoad – Movesense IMU Web Monitor & Jump Detection

vLoad is a FastAPI-based web application for **real-time monitoring of Movesense IMU sensors** and **figure-skating jump detection** based on IMU9 data. It combines BLE streaming, video recording, IMU-based jump detection, and video clip extraction into a single workflow.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Server (server.py)                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  AppState (app_state.py) – single source of truth for runtime state      │   │
│  │  Injected via Depends(get_state) into routes                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Lifespan starts:                                                               │
│    • init_db() (optional PostgreSQL)                                            │
│    • _jump_worker_loop (async) – consumes jump_sample_queue, runs JumpDetector  │
│    • _frame_sync_loop (async) – syncs video frames → frame_history              │
│    • UDP receiver (localhost) – receives IMU packets from imu_collector         │
│    • jump_clip_worker subprocess (file-queue) – cuts per-jump MP4 clips         │
│    • _auto_connect_loop – connects IMU + video for default skater if configured │
└─────────────────────────────────────────────────────────────────────────────────┘
         │                    │                     │
         ▼                    ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ imu_collector    │  │ video_backend    │  │ jump_clip_worker │
│ (subprocess)     │  │ (Picamera2/etc)  │  │ (subprocess)     │
│ Bleak → Movesense│  │ Records to       │  │ Polls jobs dir,  │
│ → UDP packets    │  │ session dir      │  │ cuts clips       │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                    │                     │
         ▼                    ▼                     ▼
    UDP localhost        frame_history         data/jobs/jump_clips
    (127.0.0.1:9999)     (deque)               *.json → done/failed
```

---

## Directory Structure

| Path | Purpose |
|------|---------|
| `server.py` | FastAPI app, lifespan, main routes (/connect, /detection/*, /annotations/*, /config, /files) |
| `app_state.py` | AppState class – session, buffers, queues, process refs |
| `deps.py` | FastAPI dependency `get_state` |
| `schemas/` | Pydantic request/response models (ConnectPayload, SessionStartPayload, etc.) |
| `routers/` | Route handlers grouped by domain |
| `modules/` | Core logic (BLE, jump detection, video, DB) |
| `UI/` | HTML templates, static JS/CSS |
| `data/sessions/<id>/` | Session dir: video.mp4, frames.csv, imu.csv, jump_clips/ |
| `data/jobs/jump_clips/` | Clip job queue: *.json → done/*.done.json, failed/*.failed.json |

---

## Workflow (Step by Step)

### 1. Startup

- **Lifespan** (`server.py`): `init_db()` if `config.json` has `database.url`; create AppState; start `_jump_worker_loop`, `_frame_sync_loop`, UDP receiver, jump_clip_worker subprocess, video backend.
- **Auto-connect** (optional): If a default skater exists and has a registered device, start IMU collector and video; retry IMU until data arrives.

### 2. Connect IMU

- **POST /connect** (`server.py`): Accepts `device` (MAC/name) or `skater_id`. Spawns `modules.imu_collector` subprocess:
  - Connects to Movesense via Bleak (`modules/movesense_gatt.py`)
  - Subscribes to IMU6/IMU9 at configured rate
  - Sends JSON packets over UDP to server

### 3. UDP → Server

- **`_ImuUdpProtocol.datagram_received`**:
  - `type: "imu"` → push samples to `jump_sample_queue`, `imu_history`; write to session `imu.csv`; broadcast to WebSocket
  - `type: "log"` → forward to WebSocket
  - `type: "collector_stat"` → update `st.dbg`

### 4. Jump Detection

- **`_jump_worker_loop`**:
  - Consumes `jump_sample_queue`
  - Feeds samples to `JumpDetectorRealtime` (`modules/jump_detector.py`)
  - Uses `web_jump_detection.py`: preprocess → peaks → windows → metrics → `select_jump_events`
  - Emits jump events only when `jump_detection_enabled` (POST /detection/start)
  - For each event: broadcast to WebSocket; persist to DB; enqueue clip job

### 5. Recording

- **POST /session/start** (`routers/sessions.py`):
  - Create session dir under `data/sessions/<session_id>/`
  - Start video backend recording; open `imu.csv`
  - `_frame_sync_loop` copies frames from video backend to `frame_history`

- **POST /session/stop**:
  - Stop recording; mux H264 → MP4; close IMU CSV; persist frames to DB

### 6. Clip Generation

- On jump detection, server writes a job JSON to `data/jobs/jump_clips/jump_<id>_<ts>.json`
- **`jump_clip_worker`** (`modules/jump_clip_worker.py`):
  - Polls jobs dir; processes each job
  - Cuts clip from session `video.mp4` (or `video.h264`) using host-time window
  - Writes `jump_clips/jump_<event_id>.mp4`; updates DB `video_path`; extracts frame times

### 7. Jump Review (UI)

- **GET /jumps** → SPA shell; client loads `/api/fragments/jumps` with preloaded jump list
- **GET /db/jumps** → list jumps from DB
- **GET /db/jumps/{jump_id}** → full jump row with IMU samples, clip path
- **POST /db/jumps/{jump_id}/marks** → store video-verified takeoff/landing marks
- **POST /pose/jumps/{jump_id}/run** → run MediaPipe pose on clip, update pose metrics

---

## Code Function Walkthrough

### Entry Point

- **`server.py`**: FastAPI app with `lifespan` context manager. Mounts routers: pages, api_devices, api_skaters, api_coaches, api_jumps, video, sessions, ws. Defines `/connect`, `/disconnect`, `/detection/start`, `/detection/stop`, `/annotations/*`, `/config`, `/files`, `/export`.

### State & Dependencies

- **`app_state.py`**: `AppState` holds session_id, session_dir, frame_history, imu_history, jump_sample_queue, jump_events, jump_annotations, jump_windows_by_session, dbg, and refs to processes/tasks.
- **`deps.py`**: `get_state(request)` returns `request.app.state.state` (AppState).

### Routers

| Router | Routes | Purpose |
|--------|--------|---------|
| `routers/pages.py` | `/`, `/jumps`, `/devices`, `/skaters`, `/coaches`; `/api/fragments/*` | SPA shell + fragment HTML (server-rendered preload) |
| `routers/ws.py` | `WebSocket /ws` | ConnectionManager; broadcast JSON to clients |
| `routers/sessions.py` | `/session/start`, `/session/stop`, `/session/status`; `/sessions/{id}/video`, `/sessions/{id}/frames`; `/debug/status` | Recording lifecycle, frame/video access |
| `routers/video.py` | `/video/connect`, `/video/disconnect`, `/video/status`, `/video/mjpeg`, `/video/snapshot.jpg` | Video backend control and streams |
| `routers/api_devices.py` | `/scan`, `/api/devices`, `/api/devices/{id}` | BLE scan (MovesenseGATTClient), device CRUD |
| `routers/api_skaters.py` | `/api/skaters`, `/api/skaters/default`, `/api/skaters/{id}` | Skater CRUD, default skater |
| `routers/api_coaches.py` | `/api/coaches`, `/api/skaters/{id}/coaches`, etc. | Coach CRUD, skater–coach links |
| `routers/api_jumps.py` | `/db/jumps`, `/db/jumps/{jump_id}`, `/db/jumps/{id}/marks`, `/pose/jumps/{id}/run`, bulk_delete | Jump CRUD, marks, pose run |

### Modules

| Module | Purpose |
|--------|---------|
| `modules/config.py` | Load `config.json`; `AppConfig` dataclasses (database, movesense, imu_udp, sessions, video, jump_recording) |
| `modules/movesense_gatt.py` | `MovesenseGATTClient` – Bleak wrapper; scan, connect, subscribe_imu; GSP decode |
| `modules/imu_collector.py` | Standalone process: connects to Movesense, subscribes IMU, timestamps samples, sends UDP packets |
| `modules/jump_detector.py` | `JumpDetectorRealtime` – rolling buffer; calls `web_jump_detection` for preprocessing, peaks, windows, metrics, event selection |
| `modules/web_jump_detection.py` | `preprocess_vertical_series`, `find_vertical_peaks`, `build_jump_windows`, `compute_window_metrics`, `select_jump_events` (Bruening-style) |
| `modules/jump_clip_worker.py` | Subprocess: polls job files; cuts clips from session video; updates DB `video_path`; extracts frame times |
| `modules/video_backend.py` | `VideoBackend` ABC; `get_video_backend()` returns Picamera2/Jetson/HttpProxy; `get_video_backend_for_tools()` returns FfmpegToolsBackend |
| `modules/video_backends/*` | Picamera2Backend, JetsonGStreamerBackend, HttpProxyVideoBackend, FfmpegToolsBackend |
| `modules/video_tools.py` | `cut_h264_clip_to_mp4_best_effort`, `extract_mp4_frame_times`, `probe_mp4_stream_info` |
| `modules/db/*` | Pool (init_db, get_pool), sessions, jumps, skaters, coaches, devices; PostgreSQL via asyncpg |

### Schemas

- **`schemas/requests.py`**: `ConnectPayload` (device, skater_id, mode, rate), `SessionStartPayload`, `JumpMarksPayload`, `BulkDeletePayload`.

---

## API Summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | SPA shell (Connect) |
| GET | `/jumps`, `/devices`, `/skaters`, `/coaches` | SPA shell for each view |
| GET | `/api/fragments/connect`, `/jumps`, … | Full page HTML for SPA |
| WebSocket | `/ws` | Live IMU, logs, jump events |
| POST | `/connect` | Start IMU collector (device or skater_id) |
| POST | `/disconnect` | Stop IMU collector |
| POST | `/detection/start` | Enable jump detection (auto-start session) |
| POST | `/detection/stop` | Disable jump detection |
| POST | `/session/start` | Start recording |
| POST | `/session/stop` | Stop recording |
| GET | `/session/status` | Current session id/dir |
| GET | `/sessions/{id}/video` | Session video file |
| GET | `/sessions/{id}/frames` | Frame timing JSON |
| GET | `/scan` | BLE scan |
| GET | `/api/devices`, `/api/skaters`, `/api/coaches` | List entities |
| GET | `/db/jumps` | List jumps |
| GET | `/db/jumps/{jump_id}` | Jump detail |
| POST | `/db/jumps/{jump_id}/marks` | Store takeoff/landing marks |
| POST | `/pose/jumps/{jump_id}/run` | Run pose analysis |
| GET | `/files` | Serve file (data/ subtree) |
| GET | `/config`, POST `/config` | Read/write config |
| GET | `/export` | Export IMU history |
| GET | `/debug/status` | Diagnostics |

---

## Configuration

Runtime settings are read from **`config.json`** in the repo root. Copy `config.example.json` to `config.json` and edit.

| Section | Key | Description |
|---------|-----|-------------|
| `database` | `url` | PostgreSQL URL; empty = persistence disabled |
| `movesense` | `default_device`, `default_mode`, `default_rate`, `ble_adapter` | IMU defaults |
| `imu_udp` | `host`, `port` | Where imu_collector sends packets (default 127.0.0.1:9999) |
| `auto_connect` | `imu_retry_interval_seconds` | Retry interval for auto IMU connect |
| `jobs` | `jump_clip_jobs_dir` | Clip job queue directory |
| `sessions` | `base_dir`, `jump_clips_subdir` | Session storage |
| `video` | `backend` | picamera2, jetson |
| `jump_recording` | `pre_jump_seconds`, `post_jump_seconds`, `clip_buffer_seconds` | Window around detected jump |

---

## Running the App

```bash
cd ~/vLoad
source venv/bin/activate   # or .\venv\Scripts\activate on Windows

python -m uvicorn server:app --host 0.0.0.0 --port 8080
```

Open `http://<server-ip>:8080`. Connect page: select skater, click Connect. Enable detection when ready; start/stop session for recording.

---

## Raspberry Pi: USB Bluetooth Adapter

This project uses **Bleak** on Linux with **BlueZ**. If you disable onboard Bluetooth, a USB adapter typically becomes `hci0`.

### Disable onboard Bluetooth

1. Edit boot config: Raspberry Pi OS Bookworm `/boot/firmware/config.txt`; older images `/boot/config.txt`
2. Add: `dtoverlay=disable-bt`
3. Run: `sudo systemctl disable --now hciuart`
4. Reboot: `sudo reboot`

### Verify USB adapter

```bash
lsusb
bluetoothctl list
bluetoothctl show
```

To force a specific adapter (e.g. `hci1`), set in `config.json`:

```json
{ "movesense": { "ble_adapter": "hci1" } }
```
