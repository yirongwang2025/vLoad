## Project Plan – Enhanced Jump Analysis (12/20/2025)

This plan updates `project.plan.12.14.2025.md` after reviewing the PDFs in `Papers/` and comparing their methods to the current `vLoad` implementation.

---

## 1) Papers reviewed (high-signal takeaways)

### 1.1 Bruening et al., 2018 (figure skating, waist IMU)
**Goal:** count multi‑rev jumps, estimate height from flight time, and quantify rotation speed using a waist IMU.

**Jump identification (their core detector):**
- Find **two vertical acceleration peaks** (take‑off + landing), both **> 25 m/s²**.
- Peak pair must be separated by **0.3–0.85 s**.
- Require a **rotation (gyro) peak > 688 °/s (12 rad/s)** between peaks.

**Height / flight time (they evaluated multiple flight-time estimators):**
- Uses \(h = gT_f^2/8\) but emphasizes that **peaks occur before/after true liftoff/landing**, so time-between-peaks needs correction.
- Tested multiple flight time algorithms including:
  - **Gravitational threshold** at ~**1.5 g**
  - **Peak-to-peak scaling** (their best MAE; tuned scale factor ~0.761 on their dataset)
  - **Valley-to-valley scaling**
  - **Vertical/horizontal acceleration intersection**

**Rotation speed:**
- Peak \( \omega_z \) occurs at ~**0.64 ± 0.16** of the interval between the two accel peaks.

### 1.2 Jones et al., 2022 (figure skating, waist IMU) – take‑off classification
**Goal:** classify take‑off type **edge vs toe** after jumps have been detected.
- Build one **input vector per jump** centered on **take‑off time** with **60 samples before/after**.
- Compress / preserve peaks by retaining the **largest reading per adjacent pair**.
- Trained **AdaBoost (scikit‑learn)**, evaluated with **leave-one-athlete-out** cross validation.
- Achieved **F1 ≈ 0.92** for edge vs toe classification.

### 1.3 Panfili et al., 2022 (inline figure skating) – foot distance sensors + back gyro
**Goal:** robust jump detection and height estimation during practice.
- Uses **under‑foot vertical distance sensors** + **back IMU**.
- Detection is staged: **threshold crossing several consecutive samples** (e.g., N=3) within time limits, then **gyro-based filtering** to reduce false positives.
- Key idea: **direct measurement of foot-to-ground distance** makes segmentation and height less ambiguous than IMU-only peak timing.

### 1.4 Harding et al., 2008 (snowboarding aerials) – rotation classification via gyro integration
**Goal:** classify rotation group (e.g., 180/360/540/720) from body gyro.
- **Integrate angular velocity over airtime** (“integration by summation”) to get angular displacement.
- Use integrated displacement (and/or composite “Air Angle”) to classify rotational complexity.

### 1.5 Caine et al., 2010 (injury epidemiology)
**Relevance:** strong motivation for tracking **training volume**, impacts, and fatigue proxies; not an algorithm paper.

### 1.6 Pedregosa et al., 2011 (scikit‑learn)
**Relevance:** provides patterns for implementing the classification stack cleanly:
- pipelines, cross-validation, reproducibility, consistent APIs.

---

## 2) Where we are today (current repo behavior)

### 2.1 Current detection & metrics (implemented)
- **Vertical pipeline**: `az_no_g_smooth` peak detection → candidate windows → metrics:
  - `height` from \(h = gT_f^2/8\)
  - `peak_gz`, `t_peak_gz`, `rotation_phase`
  - `confidence` heuristic
- **2.1 (segmentation refinement)**: take‑off/landing refined via **sign crossings** of `az_no_g_smooth`.
- **2.2 (revolutions/UR)**: bias-correct `gz_smooth` (median pre‑takeoff) and **integrate** over flight window to estimate:
  - `revolutions_est`, `revolutions_class`, `underrotation`, `underrot_flag`
- **2.3 (sanity/logging)**:
  - `[Jump] ... rev≈..., UR=...`
  - `min_revs` config to filter emitted events.

---

## 3) Comparison vs papers (what matches, what’s missing)

### 3.1 Matches (good alignment)
- **Bruening-style structure**: detect candidate events from accel peaks + constrain by flight-time + require rotation.
- **Rotation peak timing**: we compute `t_peak_gz` and `rotation_phase`, consistent with Bruening’s “peak mid-flight” observation.
- **Revolution estimation**: our Step 2.2 is conceptually aligned with Harding’s “gyro integration” approach.
- **Classification direction**: Jones’ “detect jump → classify take-off type” fits our planned architecture.

### 3.2 Gaps / risks (high priority)

#### A) **Timestamp correctness (CRITICAL for rev + flight-time accuracy)**
Today, `ble_worker` assigns the **same `t`** to all decoded samples in a BLE notification burst (8 samples). That causes:
- near‑zero `dt` → unstable smoothing / integration,
- biased `flight_time` and especially **broken revolution integration** (trapz area collapses when dt≤0).

**Plan:** derive per-sample timestamps using:
- device timestamp (`imu_timestamp`) + sample rate, or
- wall-clock + sample index offsets (`t + i / rate`), at minimum.

#### B) **Axis/vertical assumptions**
Bruening uses “vertical” acceleration and rotation, but in practice:
- waist IMU orientation varies across sessions, belt rotation, lean angles, etc.

**Plan:** add sensor fusion (IMU9) to estimate gravity vector and express:
- vertical acceleration in a gravity-aligned frame,
- yaw rate in a consistent frame for more stable revolution estimates.

#### C) **Flight-time estimator vs Bruening “scaling”**
Bruening found peak-to-peak scaling worked best (tuned factor). We currently:
- use peak pairing + sign-crossing refinement (no learned scaling).

**Plan:** evaluate on labeled data whether a simple **calibration factor** or per-skater factor improves height MAE.

#### D) **False positives / confounds**
Bruening’s thresholds are much higher than ours (their peaks > 25 m/s² and gyro peak > 688 °/s).
Our current thresholds are intentionally permissive; this may increase false positives in real programs.

**Plan:** data-driven threshold tuning + optional multi-stage filtering (similar in spirit to Panfili’s staged approach).

---

## 4) Proposed implementation plan (phased)

### Phase A — Fix foundations (data correctness)
**A1. Per-sample timestamping**
- Update BLE decode path to assign each sample an accurate timestamp.
- Persist both `t` and `imu_timestamp` per sample.

**Acceptance criteria**
- `t` is strictly increasing within each 8-sample packet.
- `revolutions_est` is non-zero for real multi-rev jumps and stable across repeats.

**A2. Window extraction and storage correctness**
- Ensure `/export?mode=jumps` window samples include time-aligned data and any needed fields for offline processing.

---

### Phase A3 — OAK‑D live video streaming + synchronized recording (enables Phase B evaluation)
**Why this is needed:** Phase B compares segmentation algorithms against **video ground truth** (Bruening-style \(t_{takeoff}^{GT}, t_{landing}^{GT}\)). We already stream IMU; we need a repeatable way to **stream + record OAK‑D video alongside IMU with aligned timestamps**.

#### A3.1 Goals
- **Live view** of OAK‑D RGB (and optionally depth) in the browser during sessions.
- **Session recording** of:
  - video (RGB; optional depth),
  - IMU samples (already available),
  - per-frame/per-sample timestamps in a single session folder.
- A synchronization approach that is **good enough for labeling** take-off/landing to ±1–2 frames, with a path to improve later.

#### A3.2 Recommended architecture (minimal, robust)
- **Backend**: use Luxonis **DepthAI** Python SDK (`depthai`) to pull frames from OAK‑D.
- **Streaming format (MVP)**: HTTP **MJPEG** stream:
  - endpoint: `GET /video/mjpeg`
  - browser: `<img src="/video/mjpeg">`
  - pros: simple, works everywhere, low integration friction.
  - cons: no audio, higher bandwidth than modern codecs; acceptable for MVP.
- **Recording (MVP)**:
  - encode frames to **MP4 (H.264)** via either:
    - DepthAI encoder node (preferred), or
    - OpenCV `VideoWriter` (fallback).
  - write a **`frames.csv`** with per-frame timestamps.

#### A3.3 Timestamp & synchronization strategy (practical now, improvable later)
We will standardize on **host wall-clock epoch seconds** (`time.time()`) as the cross-modal reference:
- For each **video frame**, record:
  - `t_host`: host epoch seconds at frame arrival/processing,
  - `oak_timestamp`: device timestamp if available (DepthAI provides per-frame timestamps),
  - `frame_idx`.
- For each **IMU sample**, we already store:
  - `t` (host epoch seconds), plus `imu_timestamp` and `imu_sample_index`.

**Sync quality improvement knobs:**
- **MVP “visual sync cue”**: add a UI button “Mark Sync” that logs a server timestamp, while the user performs a visible gesture in camera view (e.g., sharp hand clap / flashlight blink). This gives a clear anchor for aligning streams during labeling.
- **Next step (optional)**: use DepthAI device timestamps + host offset estimation for tighter alignment (reduces jitter vs host capture time).
- **Longer term**: hardware trigger sync (if available) between IMU and camera; not required for Phase B MVP.

#### A3.4 Session & file schema (so labeling is easy)
Create a session directory per recording:
```
data/sessions/<session_id>/
  session.json
  imu.csv                # per-sample IMU (t, imu_timestamp, imu_sample_index, acc, gyro, mag)
  video.mp4              # RGB recording (or rgb.mp4)
  frames.csv             # frame_idx, t_host, oak_timestamp (optional), width, height
  events.jsonl           # sync markers and user annotations (optional)
```
Where:
- `session_id` = timestamp + short suffix.
- `session.json` stores config snapshot (`rate`, `mode`, jump config thresholds, min_revs, camera settings).

#### A3.5 Backend implementation tasks
- **New module**: `modules/oakd_stream.py`
  - manage a single camera pipeline (RGB only MVP),
  - provide:
    - `start_camera()`, `stop_camera()`,
    - `get_latest_jpeg()` (for MJPEG streaming),
    - `start_recording(session_id)`, `stop_recording()`.
- **Server endpoints (FastAPI)**:
  - `GET /video/mjpeg` (multipart/x-mixed-replace)
  - `POST /video/start` / `POST /video/stop` (camera lifecycle)
  - `POST /session/start` / `POST /session/stop`
    - starts camera recording + IMU capture to `data/sessions/<id>/...`
  - `POST /session/mark_sync` (writes a sync marker with server time)
- **Concurrency**:
  - camera capture runs in a background task/thread (DepthAI callback or polling),
  - latest-frame buffer protected by a lock,
  - recording writer decoupled (queue) so streaming stays smooth.

#### A3.6 Frontend (UI) changes (MVP)
- Add a “Video” panel/tab (likely in `UI/jumps.html` first):
  - `<img id="videoFeed" src="/video/mjpeg">`
  - buttons:
    - Start/Stop Session
    - Mark Sync
  - show current session_id and basic stats (fps, dropped frames).

#### A3.7 Acceptance criteria (to unblock Phase B)
- Video appears in browser with stable FPS (target 15–30 fps).
- During a session, we produce:
  - a playable `video.mp4`,
  - `imu.csv` with monotonically increasing `t`,
  - `frames.csv` with monotonically increasing `t_host`.
- We can label ~10 jumps with take-off/landing times using the recorded video and map them to IMU windows via timestamps.

---

### Phase B — Calibrated segmentation + height accuracy
**B1. Evaluate segmentation refinement vs Bruening**
- Compare:
  - our sign-crossing approach,
  - Bruening GT (1.5g threshold),
  - peak-to-peak and valley-to-valley scaling (with learned constants).

**B2. Height calibration**
- Add a configurable `height_scale` (global or per-skater) if it improves MAE.

**Acceptance criteria**
- Flight time MAE and height MAE decrease on a labeled dataset (video/OAK-D).

---

### Phase C — Revolution counting robustness (Step 2.2 upgrade)
**C1. Bias estimation improvements**
- Median window is a good start; add robustness:
  - reject windows with motion (high variance),
  - expand/shrink baseline based on signal quality.

**C2. Frame correction**
- Use IMU9 fusion to integrate yaw about gravity rather than raw sensor `gz`.

**Acceptance criteria**
- Revolution estimate error within a target band on labeled clips (e.g., mean abs error < 0.25 rev).

---

### Phase D — Jump type classification (Jones-aligned)
**D1. Dataset + labels**
- Export per-jump windows (already supported) + build label file:
  - take-off type (edge vs toe),
  - jump family (Sal/Toe/Loop/Flip/Lutz/Axel),
  - revolutions_true + UR info.

**D2. Baseline classifier**
- Implement a scikit-learn training script:
  - feature vectors centered on `t_takeoff` (Jones: 60 samples pre/post),
  - leave-one-skater-out validation,
  - start with AdaBoost, then compare to linear models / trees.

**D3. Realtime inference**
- Run inference on each detected event and attach predicted labels to WebSocket payload and export.

**Acceptance criteria**
- Edge vs toe F1 ≥ 0.90 on cross-validation (or best achievable with our sensor/video setup).

---

## 5) Immediate next tasks (recommended order)
1. **Fix per-sample timestamps** in the BLE ingest pipeline (required for everything downstream).
2. Build a small labeled validation set (10–20 jumps) using OAK‑D/video timing.
3. Compare flight-time estimators + consider a light scaling/calibration option (Bruening).
4. Upgrade revolution counting to gravity-aligned yaw integration (Harding-inspired but IMU9-correct).
5. Add Jones-style take-off classifier pipeline (scikit-learn).

---

## 6) Notes on repo artifacts
- `extract_papers.py` and `Papers/_extracted_text/*.txt` were used to extract and review text from PDFs.
  - If desired, we can add a `.gitignore` entry later to avoid checking extracted text into the repo.


