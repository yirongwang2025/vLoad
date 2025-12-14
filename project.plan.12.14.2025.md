## Project Plan – IMU‑Based Figure Skating Jump Detection & Classification  
**Date:** 2025‑12‑14  

This document summarizes the next phases of work to extend the existing Bruening‑style IMU jump detector into a system that (1) robustly detects multi‑rev jumps, (2) estimates flight time, height and revolutions, and (3) classifies jump type (Sal, Toe, Loop, Flip, Lutz, Axel, etc.) using methods inspired by Bruening et al. 2018, Panfili et al. 2022, Jones et al. 2022, and related IMU/ML work (e.g., aerial acrobatics classification, scikit‑learn design patterns).  

---

## 0. Current Baseline (As of 2025‑12‑14)

- **Hardware & placement**
  - Single Movesense IMU9 at waist/lower‑back (Bruening‑style).
  - OAK‑D S2 stereo RGB‑D camera rink‑side, providing synchronized monocular + depth video for offline labelling and, later, IMU/video fusion.
  - BLE streaming from the IMU to `server.py` via `modules/movesense_gatt.py`.

- **Backend pipeline**
  - `MovesenseGATTClient.decode_imu_payload` parses GSP IMU9 frames into 8×(acc, gyro, mag) samples.
  - `ble_worker`:
    - Connects to sensor, subscribes to IMU, pushes samples into `_jump_sample_queue` and maintains an IMU history buffer.
    - Broadcasts per‑frame summary JSON (`mode`, `rate`, `timestamp`, `first_sample`, small `analysis`).
  - `JumpDetectorRealtime` (`modules/jump_detector.py`) + `modules/web_jump_detection.py`:
    - **Step 2.1**: `preprocess_vertical_series` builds `t`, `az`, `az_no_g`, `gz` + smoothed variants (`*_smooth`) with one‑pole low‑pass (~10 Hz).
    - **Step 2.2**: `find_vertical_peaks` on |`az_no_g_smooth`|.
    - **Step 2.3**: `build_jump_windows` for plausible flight times (0.25–0.8 s).
    - **Step 2.4**: `compute_window_metrics` → `height`, `peak_az_no_g`, `peak_gz`, `t_peak_gz`.
    - **Step 2.5/2.6**: `select_jump_events` with thresholding + confidence in [0,1] using normalised height / |az−g| / |ωz| and rotation phase.
    - `JumpDetectorRealtime.update` logs `[Phase1]`, `[Phase2.x]`, `[Jump] … h=…, ωz=…, phase=…, conf=…` and returns `new_events`.
  - `_jump_worker_loop`:
    - Consumes samples from `_jump_sample_queue`, runs `JumpDetectorRealtime.update`, and emits `{"type": "jump", …}` over WebSocket once jump detection is enabled.

- **Frontend (inline `INDEX_HTML` in `server.py`)**
  - Live plots for accel / gyro / mag.
  - Jump detection settings (min height, min |az−g|, min ωz, min separation) editable via `/config`.
  - Data export (`/export`), log window.
  - **Jump events panel**: shows `Total jumps` counter and a scrollable list; `ws.onmessage` handles `msg.type === 'jump'` and `msg.type === 'log'`.
  - WebSocket connect/disconnect/error logging.

This baseline covers detection, height & rotation metrics and basic confidence scoring, but does **not yet** classify jump type or compute revolutions in a principled way. We deliberately use only a single waist IMU and **no foot distance sensors** (unlike Panfili); ground‑truth and future fusion signals will instead come from the OAK‑D S2 camera (RGB + stereo depth).

---

## 1. Data & Labelling Pipeline

**Goal:** Build a high‑quality labelled dataset of jumps with ground‑truth jump type and revolution count to support supervised learning (Jones‑style take‑off classification and rev estimation).

### 1.1 Define event schema

- For each jump event, record:
  - `event_id`, `session_id`, `skater_id`, `timestamp` (e.g., `t_peak` or `t_takeoff`).
  - `jump_family` ∈ {Waltz, Sal, Toe, Loop, Flip, Lutz, Axel, Euler, Other}.
  - `revolutions_true` (e.g. 1, 2, 3, 4 or 2.5).
  - `underrotation_flag` (boolean) and `underrotation_amount` (rev short).
  - `success_flag` (landed vs fall / step‑out), optional GOE.

### 1.2 Export and link IMU + labels

- Extend or reuse `/export` to support:
  - Export of **per‑jump windows** (see §2) keyed by `event_id`.
  - A companion CSV/JSON file containing the labels above.
- Build a small annotation tool (can be a Jupyter notebook or simple script) that:
  - Loads exported IMU windows and associated OAK‑D S2 video clips from training sessions.
  - Allows manual assignment of `jump_family`, `revolutions_true`, and under‑rotation.

**Dependencies:** none (uses existing IMU export + log timestamps).  

---

## 2. Improved Event Segmentation & Revolution Counting

**Goal:** Produce accurate, repeatable estimates of take‑off/landing times and revolutions for each detected jump, in line with Bruening 2018 and UR‑monitor style work.

### 2.1 Refine take‑off / landing detection

- In `compute_window_metrics` / `JumpDetectorRealtime`:
  - Expose `t_takeoff` and `t_landing` explicitly for each window.
  - Optionally refine these times by:
    - Searching for local extrema / zero‑crossings of `az_no_g_smooth` around the coarse peak indices.
    - Optionally use `gz_smooth` to ensure the main rotation lies between `t_takeoff` and `t_landing`.

### 2.2 Compute revolution count and under‑rotation

- For each window (take‑off \(t_0\), landing \(t_1\)):
  - Compute bias‑corrected vertical angular velocity:
    - Estimate baseline bias `gz_bias` as median `gz_smooth` over a pre‑take‑off window (e.g. \([-0.5, -0.1]\) s).
    - `gz_corr = gz_smooth − gz_bias`.
  - Integrate to angle:
    \[
    \theta_z = \int_{t_0}^{t_1} \mathrm{gz\_corr}(t)\,dt
    \]
  - Compute:
    - `revolutions_est = theta_z / (2π)`.
    - `revolutions_class = round(revolutions_est)`.
    - `underrotation = revolutions_class − revolutions_est`.
    - `underrot_flag` if `underrotation < -0.25` (or tuned threshold).
- Add these fields to the metric dict and propagate them into `select_jump_events` and the final event payload (`JumpDetectorRealtime` + WebSocket `type: "jump"` messages).

### 2.3 Sanity checks & logging

- Extend `[JumpDetector] [Jump]` log to include `rev≈…`, `UR=…` to visually verify against video.
- Add options in the UI config:
  - `min_revs` (e.g. only treat events with `revolutions_class ≥ 1.5` as real jumps).

**Dependencies:** Existing `compute_window_metrics`, `select_jump_events`, new integration helpers.  

---

## 3. Feature Engineering for Jump‑Type & Rev Classification


