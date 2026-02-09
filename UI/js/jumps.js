window.JumpsPage = {
  init: function() {
const root = document.querySelector('[data-page="jumps"]');
if (!root) return;

const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
const logBox = root.querySelector('#logBox');
const jumpListEl = root.querySelector('#jumpList');
const refreshBtn = root.querySelector('#refreshBtn');
const dbStatus = root.querySelector('#dbStatus');

const jumpNameInput = root.querySelector('#jumpNameInput');
const jumpNote = root.querySelector('#jumpNote');
const saveAnnotationBtn = root.querySelector('#saveAnnotationBtn');
const annotationStatus = root.querySelector('#annotationStatus');
const deleteJumpBtn = root.querySelector('#deleteJumpBtn');
const deleteStatus = root.querySelector('#deleteStatus');

const videoPlayer = root.querySelector('#videoPlayer');
if (videoPlayer) {
  videoPlayer.addEventListener('error', async () => {
    try {
      const src = videoPlayer.currentSrc || videoPlayer.src || '';
      if (!src) return;
      // Try to surface the real HTTP error (404/500) behind "NotSupportedError"
      const resp = await fetch(src, { method: 'HEAD' });
      if (!resp.ok) {
        addLog(`Video element error; source not available (HTTP ${resp.status}).`);
        if (playerStatus) playerStatus.textContent = `Video error (HTTP ${resp.status})`;
      } else {
        addLog('Video element error (source exists). Browser may not support the codec/container.');
        if (playerStatus) playerStatus.textContent = 'Video error (codec/container not supported?)';
      }
    } catch (e) {
      addLog('Video element error; failed to diagnose: ' + e);
    }
  });
}
const playBtn = root.querySelector('#playBtn');
const playHalfBtn = root.querySelector('#playHalfBtn');
const playQuarterBtn = root.querySelector('#playQuarterBtn');
const markStartBtn = root.querySelector('#markStartBtn');
const markEndBtn = root.querySelector('#markEndBtn');
const recomputeBtn = root.querySelector('#recomputeBtn');
const prevFrameBtn = root.querySelector('#prevFrameBtn');
const nextFrameBtn = root.querySelector('#nextFrameBtn');
const stopBtn = root.querySelector('#stopBtn');
const backBtn = root.querySelector('#backBtn');
const playerStatus = root.querySelector('#playerStatus');

const canvasAcc = root.querySelector('#plotAcc');
const canvasGyro = root.querySelector('#plotGyro');
const canvasMag = root.querySelector('#plotMag');
const ctxAcc = canvasAcc ? canvasAcc.getContext('2d') : null;
const ctxGyro = canvasGyro ? canvasGyro.getContext('2d') : null;
const ctxMag = canvasMag ? canvasMag.getContext('2d') : null;

const maxJumpItems = 200;
const jumps = [];  // list rows from /db/jumps
let selectedIndex = -1;
let selectionAnchorIndex = -1; // for shift-click range
let selectedJump = null; // detail from /db/jumps/{jump_id}
const selectedJumpIds = new Set(); // Track multiple selected jumps by jump_id
let videoStartHost = null; // host epoch seconds for video t=0
let playbackRAF = null;
let playbackVfcHandle = null; // requestVideoFrameCallback handle (if supported)
let sessionFrames = null; // array of frames for current session (optional)
let sessionFps = 30; // recording fps (used to map mediaTime -> frame_idx)

// Unified host-time timeline (single source of truth for plot cursor + video position)
let _cursorHost = null; // current cursor host epoch seconds
let _timelineMode = null; // 'host_pre_video' | 'video'
let _timelineRate = 1.0;
let _timelineWallT0 = null; // performance.now() at start (host-driven phase)
let _timelineHostT0 = null; // host epoch seconds at start (host-driven phase)
let _timelineVideoStarted = false;
let _boundsCache = null; // {plotStart,plotEnd,videoStart,videoEnd,globalStart,globalEnd}
let _unifiedActive = false; // true when playback started via unified timeline (Play / slow play)

// Smooth plot cursor during video playback: interpolate host time between decoded video frames
// so the cursor moves smoothly (instead of stepping only when frames advance).
let _smoothCursorRAF = null;
let _smoothAnchorWall = null; // performance.now()/Date.now at last anchor
let _smoothAnchorHost = null; // host epoch seconds at last anchor (from decoded frame)

// Plot buffers
const maxPts = 150;
const accX = [], accY = [], accZ = [];
const gyroX = [], gyroY = [], gyroZ = [];
const magX = [], magY = [], magZ = [];
const colors = ['#1976d2', '#d32f2f', '#388e3c'];
let _plotT0 = null; // host epoch seconds (plot start)
let _plotT1 = null; // host epoch seconds (plot end)

function addLog(line) {
  if (!logBox) return;
  try {
    const ts = new Date().toISOString();
    logBox.textContent += `[${ts}] ${line}\n`;
    logBox.scrollTop = logBox.scrollHeight;
  } catch (e) { /* avoid breaking if log fails */ }
}

ws.onopen = () => addLog('WebSocket connected');
ws.onclose = () => addLog('WebSocket disconnected');
ws.onerror = () => addLog('WebSocket error');
ws.onmessage = (ev) => {
  try {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'log' && typeof msg.msg === 'string') addLog(msg.msg);
  } catch (e) {
    // ignore
  }
};

function fmt(num, digits) {
  if (typeof num !== 'number' || !isFinite(num)) return '';
  return num.toFixed(digits);
}

function formatTimeFromEpoch(tSec) {
  if (typeof tSec !== 'number' || !isFinite(tSec)) return '';
  const d = new Date(tSec * 1000);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  const ms = String(d.getMilliseconds()).padStart(3, '0');
  return `${hh}:${mm}:${ss}.${ms}`;
}

function pushLimited(arr, value) {
  arr.push(value);
  while (arr.length > maxPts) arr.shift();
}

function _plotMarkers() {
  // Host-time markers to remove ambiguity: all are epoch seconds.
  // Colors: peak=black, calc=blue, video marks=orange.
  if (!selectedJump) return [];
  const out = [];
  function add(t, label, color) {
    if (typeof t !== 'number' || !isFinite(t)) return;
    out.push({ t, label, color });
  }
  add(selectedJump.t_peak, 'peak', 'rgba(0,0,0,0.45)');
  add(selectedJump.t_takeoff_calc, 'calc TO', 'rgba(25,118,210,0.55)');
  add(selectedJump.t_landing_calc, 'calc LD', 'rgba(25,118,210,0.55)');
  // Video-verified marks (host time mapped from clip/session frames)
  add(selectedJump.t_takeoff_video, 'vid TO', 'rgba(245,124,0,0.60)');
  add(selectedJump.t_landing_video, 'vid LD', 'rgba(245,124,0,0.60)');
  return out;
}

function drawSeries(ctx, series, sampleRateHint, cursorFrac, timeSpanSec, t0Epoch, t1Epoch, cursorEpoch, markers) {
  const canvas = ctx.canvas;
  const h = canvas.height, w = canvas.width;
  const n = Math.max(series[0].length, series[1].length, series[2].length);
  if (n <= 1) {
    ctx.clearRect(0,0,w,h);
    return;
  }
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
  ctx.fillStyle = '#000';
  ctx.font = '10px sans-serif';
  ctx.textBaseline = 'top';
  ctx.fillText(maxV.toFixed(2), 2, 2);
  ctx.textBaseline = 'middle';
  ctx.fillText(((maxV + minV) / 2).toFixed(2), 2, h / 2);
  ctx.textBaseline = 'bottom';
  ctx.fillText(minV.toFixed(2), 2, h - 2);
  ctx.textBaseline = 'bottom';
  let timeLabel = 'Time';
  if (typeof timeSpanSec === 'number' && isFinite(timeSpanSec) && timeSpanSec > 0) {
    timeLabel += ` (~${timeSpanSec.toFixed(2)} s)`;
  } else if (sampleRateHint && n > 1) {
    const seconds = n / sampleRateHint;
    timeLabel += ` (~${seconds.toFixed(2)} s window)`;
  }
  // If we know the host epoch range, show human-readable clock times for sync debugging.
  if (typeof t0Epoch === 'number' && isFinite(t0Epoch) && typeof t1Epoch === 'number' && isFinite(t1Epoch) && t1Epoch > t0Epoch) {
    timeLabel += `  ${formatTimeFromEpoch(t0Epoch)} → ${formatTimeFromEpoch(t1Epoch)}`;
  }
  if (typeof cursorEpoch === 'number' && isFinite(cursorEpoch)) {
    timeLabel += `  @ ${formatTimeFromEpoch(cursorEpoch)}`;
  }
  const timeWidth = ctx.measureText(timeLabel).width;
  ctx.fillText(timeLabel, (w - timeWidth) / 2, h - 2);
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
  series.forEach((arr, idx) => {
    if (!arr.length) return;
    ctx.strokeStyle = colors[idx];
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = (i / Math.max(1, n - 1)) * (w - 2 * padding) + padding;
      const v = arr[i];
      const norm = (v - minV) / range;
      const y = h - (norm * (h - 2 * padding) + padding);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // Host-time markers (takeoff/landing/peak), if we know the epoch range.
  if (Array.isArray(markers) && markers.length && typeof t0Epoch === 'number' && typeof t1Epoch === 'number' && isFinite(t0Epoch) && isFinite(t1Epoch) && t1Epoch > t0Epoch) {
    ctx.save();
    ctx.lineWidth = 1;
    ctx.font = '10px sans-serif';
    ctx.textBaseline = 'top';
    for (const m of markers) {
      try {
        const tt = Number(m.t);
        if (!isFinite(tt) || tt < t0Epoch || tt > t1Epoch) continue;
        const frac = (tt - t0Epoch) / (t1Epoch - t0Epoch);
        const x = frac * (w - 2 * padding) + padding;
        ctx.strokeStyle = m.color || 'rgba(0,0,0,0.25)';
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, h - padding);
        ctx.stroke();
        // small label near bottom
        ctx.fillStyle = m.color || 'rgba(0,0,0,0.25)';
        const lbl = String(m.label || '');
        if (lbl) ctx.fillText(lbl, x + 2, h - padding - 12);
      } catch (e) {}
    }
    ctx.restore();
  }

  // Cursor marker (current video time) if provided
  if (typeof cursorFrac === 'number' && isFinite(cursorFrac)) {
    const cf = Math.max(0, Math.min(1, cursorFrac));
    const x = cf * (w - 2 * padding) + padding;
    ctx.save();
    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, padding);
    ctx.lineTo(x, h - padding);
    ctx.stroke();
    ctx.restore();
  }
}

function redrawPlots(sampleRateHint, cursorFrac, timeSpanSec, t0Epoch, t1Epoch, cursorEpoch) {
  const markers = _plotMarkers();
  if (ctxAcc) drawSeries(ctxAcc, [accX, accY, accZ], sampleRateHint, cursorFrac, timeSpanSec, t0Epoch, t1Epoch, cursorEpoch, markers);
  if (ctxGyro) drawSeries(ctxGyro, [gyroX, gyroY, gyroZ], sampleRateHint, cursorFrac, timeSpanSec, t0Epoch, t1Epoch, cursorEpoch, markers);
  if (ctxMag) drawSeries(ctxMag, [magX, magY, magZ], sampleRateHint, cursorFrac, timeSpanSec, t0Epoch, t1Epoch, cursorEpoch, markers);
}

function updateSelectionHighlight() {
  const children = jumpListEl.children;
  for (let i = 0; i < children.length; i++) {
    const jump = jumps[i];
    const jumpId = jump && typeof jump.jump_id === 'number' ? jump.jump_id : null;
    const isSelected = (i === selectedIndex) || (jumpId !== null && selectedJumpIds.has(jumpId));
    if (isSelected) children[i].classList.add('selected');
    else children[i].classList.remove('selected');
  }
}

function renderSelectedDetail() {
  const detail = root.querySelector('#jumpDetail');
  if (!detail) return;
  if (!selectedJump) {
    detail.innerHTML = '<div><em>No jump selected</em></div>';
    return;
  }
  const ev = selectedJump;
  const sid = ev.session_id || '';

  function _row(label, valueHtml, helpHtml) {
    return '<div><strong>' + label + ':</strong> ' + valueHtml + (helpHtml ? ('<div class="small" style="margin-left:10px;">' + helpHtml + '</div>') : '') + '</div>';
  }
  function _tsRow(label, tVal, extraHtml, helpHtml) {
    if (typeof tVal !== 'number' || !isFinite(tVal)) return '';
    const extra = extraHtml ? (' ' + extraHtml) : '';
    return _row(label, fmt(tVal, 3) + ' (' + formatTimeFromEpoch(tVal) + ')' + extra, helpHtml);
  }
  function _numRow(label, v, unit, helpHtml, digits=3) {
    if (typeof v !== 'number' || !isFinite(v)) return '';
    const u = unit ? (' ' + unit) : '';
    return _row(label, fmt(v, digits) + u, helpHtml);
  }
  function _boolRow(label, v, helpHtml) {
    if (typeof v !== 'boolean') return '';
    return _row(label, (v ? 'true' : 'false'), helpHtml);
  }

  function _fmtPeak(v) {
    if (typeof v !== 'number' || !isFinite(v)) return '';
    const av = Math.abs(v);
    if (av >= 1000) return av.toFixed(0);
    if (av >= 100) return av.toFixed(1);
    return av.toFixed(2);
  }

  function _imuPeakStats() {
    // Compute from FULL imu_samples (not downsampled plot) so we can verify whether
    // the biggest peaks are actually late, or whether it is a plotting/labeling issue.
    try {
      if (!ev || !Array.isArray(ev.imu_samples) || !ev.imu_samples.length) return null;

      // Cache by event_id + sample count to avoid recompute on minor UI updates.
      const cacheKey = String(ev.event_id || '') + ':' + String(ev.imu_samples.length);
      if (ev._imu_peak_cache_key === cacheKey && ev._imu_peak_cache) return ev._imu_peak_cache;

      const to = (typeof ev.t_takeoff_video === 'number' ? ev.t_takeoff_video
        : (typeof ev.t_takeoff_calc === 'number' ? ev.t_takeoff_calc : null));
      const ld = (typeof ev.t_landing_video === 'number' ? ev.t_landing_video
        : (typeof ev.t_landing_calc === 'number' ? ev.t_landing_calc : null));

      const pre0 = (typeof to === 'number' ? (to - 1.5) : null);
      const pre1 = (typeof to === 'number' ? (to - 0.1) : null);
      const flight0 = (typeof to === 'number' ? to : null);
      const flight1 = (typeof ld === 'number' ? ld : null);
      const post0 = (typeof ld === 'number' ? ld : null);
      const post1 = (typeof ld === 'number' ? (ld + 2.0) : null);

      function inWin(t, a, b) {
        if (typeof t !== 'number' || !isFinite(t)) return false;
        if (typeof a !== 'number' || !isFinite(a)) return false;
        if (typeof b !== 'number' || !isFinite(b)) return false;
        return (t >= a && t <= b);
      }

      function updPeak(obj, key, t, v) {
        const av = Math.abs(Number(v || 0));
        if (!isFinite(av)) return;
        if (obj[key] == null || av > obj[key].val) obj[key] = { t, val: av, raw: v };
      }

      const out = {
        overall: {},
        pre: {},
        flight: {},
        post: {},
        windows: { pre0, pre1, flight0, flight1, post0, post1 },
      };

      for (const s of ev.imu_samples) {
        if (!s || typeof s.t !== 'number') continue;
        const t = Number(s.t);
        const a = Array.isArray(s.acc) ? s.acc : [];
        const g = Array.isArray(s.gyro) ? s.gyro : [];
        const ax = Number(a[0] || 0), ay = Number(a[1] || 0), az = Number(a[2] || 0);
        const gx = Number(g[0] || 0), gy = Number(g[1] || 0), gz = Number(g[2] || 0);
        const aMag = Math.sqrt(ax*ax + ay*ay + az*az);
        const gMag = Math.sqrt(gx*gx + gy*gy + gz*gz);

        updPeak(out.overall, 'acc_mag', t, aMag);
        updPeak(out.overall, 'gyro_mag', t, gMag);
        updPeak(out.overall, 'acc_x', t, ax);
        updPeak(out.overall, 'acc_y', t, ay);
        updPeak(out.overall, 'acc_z', t, az);
        updPeak(out.overall, 'gyro_x', t, gx);
        updPeak(out.overall, 'gyro_y', t, gy);
        updPeak(out.overall, 'gyro_z', t, gz);

        if (inWin(t, pre0, pre1)) {
          updPeak(out.pre, 'acc_mag', t, aMag);
          updPeak(out.pre, 'gyro_mag', t, gMag);
        }
        if (inWin(t, flight0, flight1)) {
          updPeak(out.flight, 'acc_mag', t, aMag);
          updPeak(out.flight, 'gyro_mag', t, gMag);
        }
        if (inWin(t, post0, post1)) {
          updPeak(out.post, 'acc_mag', t, aMag);
          updPeak(out.post, 'gyro_mag', t, gMag);
        }
      }

      ev._imu_peak_cache_key = cacheKey;
      ev._imu_peak_cache = out;
      return out;
    } catch (e) {
      return null;
    }
  }

  const lines = [];
  lines.push(_row('jump_id', String(ev.jump_id ?? ''), 'Database primary key for this jump row (unique).'));
  lines.push(_row('event_id', String(ev.event_id ?? ''), 'Stable identifier for this detected jump event.'));
  lines.push(_tsRow('t_peak', ev.t_peak, '', 'Detection peak time (host epoch seconds) — used as the jump reference time.'));

  lines.push('<div style="margin-top:8px;"><strong>Timing</strong></div>');
  lines.push(_tsRow('calc takeoff (t_takeoff_calc)', ev.t_takeoff_calc, '', 'Algorithm-estimated takeoff time from IMU (host time).'));
  lines.push(_tsRow('calc landing (t_landing_calc)', ev.t_landing_calc, '', 'Algorithm-estimated landing time from IMU (host time).'));
  lines.push(_tsRow('video takeoff (t_takeoff_video)', ev.t_takeoff_video,
    (typeof ev.t_takeoff_video_t === 'number' ? ('<span class="small">[clip t=' + ev.t_takeoff_video_t.toFixed(3) + 's]</span>') : ''),
    'Video-verified takeoff time you marked (host time mapped from clip frames).'));
  lines.push(_tsRow('video landing (t_landing_video)', ev.t_landing_video,
    (typeof ev.t_landing_video_t === 'number' ? ('<span class="small">[clip t=' + ev.t_landing_video_t.toFixed(3) + 's]</span>') : ''),
    'Video-verified landing time you marked (host time mapped from clip frames).'));

  // IMU peak timing sanity checks (host-time only).
  try {
    const st = _imuPeakStats();
    if (st) {
      lines.push('<div style="margin-top:8px;"><strong>IMU peak timing sanity check</strong></div>');
      const oA = st.overall.acc_mag, oG = st.overall.gyro_mag;
      if (oA && typeof oA.t === 'number') {
        lines.push(_row('max |acc| (overall)', _fmtPeak(oA.val) + ' @ ' + formatTimeFromEpoch(oA.t) + ` (t=${oA.t.toFixed(3)})`,
          'Computed from raw IMU samples (vector magnitude). If this time is near the end of the plot, the peak is truly late; if not, the plot/labeling is misleading.'));
      }
      if (oG && typeof oG.t === 'number') {
        lines.push(_row('max |gyro| (overall)', _fmtPeak(oG.val) + ' @ ' + formatTimeFromEpoch(oG.t) + ` (t=${oG.t.toFixed(3)})`,
          'Computed from raw IMU samples (vector magnitude).'));
      }
      const fA = st.flight.acc_mag, fG = st.flight.gyro_mag;
      if (fA && typeof fA.t === 'number') {
        lines.push(_row('max |acc| (flight window)', _fmtPeak(fA.val) + ' @ ' + formatTimeFromEpoch(fA.t),
          'Peak IMU acceleration magnitude between takeoff→landing (prefers video marks, then calc).'));
      }
      if (fG && typeof fG.t === 'number') {
        lines.push(_row('max |gyro| (flight window)', _fmtPeak(fG.val) + ' @ ' + formatTimeFromEpoch(fG.t),
          'Peak IMU gyro magnitude between takeoff→landing (prefers video marks, then calc).'));
      }
      const pA = st.post.acc_mag, pG = st.post.gyro_mag;
      if (pA && typeof pA.t === 'number') {
        lines.push(_row('max |acc| (post window)', _fmtPeak(pA.val) + ' @ ' + formatTimeFromEpoch(pA.t),
          'Peak IMU acceleration magnitude from landing→landing+2s. If this exceeds flight peaks, it indicates large deceleration/rotation after landing.'));
      }
      if (pG && typeof pG.t === 'number') {
        lines.push(_row('max |gyro| (post window)', _fmtPeak(pG.val) + ' @ ' + formatTimeFromEpoch(pG.t),
          'Peak IMU gyro magnitude from landing→landing+2s.'));
      }
    }
  } catch (e) {}

  lines.push('<div style="margin-top:8px;"><strong>Original IMU detection metrics</strong></div>');
  lines.push(_numRow('height', ev.height, 'm', 'Estimated jump height from IMU detection window (projectile model).', 3));
  lines.push(_numRow('flight_time', ev.flight_time, 's', 'Estimated flight time from IMU detection window.', 3));
  lines.push(_numRow('theta_z_rad', ev.theta_z_rad, 'rad', 'Estimated integrated rotation (about vertical axis) from IMU gyro, over the algorithm takeoff→landing window.', 3));
  lines.push(_numRow('revolutions_est', ev.revolutions_est, 'rev', 'Estimated revolutions from IMU (|theta| / 2π).', 3));
  lines.push(_numRow('revolutions_class', ev.revolutions_class, '', 'Rounded revolutions class from IMU (e.g., 1,2,3).', 0));
  lines.push(_numRow('underrotation', ev.underrotation, 'rev', 'Revolutions_class − revolutions_est (negative means you likely rotated more than the rounded class).', 3));
  lines.push(_boolRow('underrot_flag', ev.underrot_flag, 'True if underrotation crosses a threshold (prototype).'));
  lines.push(_numRow('rotation_phase', ev.rotation_phase, '', 'When peak gyro happens within flight: (t_peak_gz − takeoff) / flight_time.', 3));
  lines.push(_numRow('acc_peak', ev.acc_peak, '', 'Peak vertical accel (no-gravity) used by detector (units depend on upstream).', 2));
  lines.push(_numRow('gyro_peak', ev.gyro_peak, '°/s', 'Peak vertical-axis gyro speed used by detector.', 0));
  lines.push(_numRow('confidence', ev.confidence, '', 'Heuristic confidence score from IMU detection.', 2));

  lines.push('<div style="margin-top:8px;"><strong>IMU metrics using your video marks</strong></div>');
  lines.push(_numRow('flight_time_marked', ev.flight_time_marked, 's', 'Computed from your Mark Start/End: (t_landing_video − t_takeoff_video).', 3));
  lines.push(_numRow('height_marked', ev.height_marked, 'm', 'Height from marked flight time using h = g*T^2/8.', 3));
  lines.push(_numRow('theta_z_rad_marked', ev.theta_z_rad_marked, 'rad', 'IMU-integrated rotation magnitude over your marked takeoff→landing interval (integral of |gyro|). Units are auto-detected (rad/s vs deg/s).', 3));
  lines.push(_numRow('revolutions_est_marked', ev.revolutions_est_marked, 'rev', 'Revolutions from IMU over marked interval.', 3));
  lines.push(_numRow('revolutions_class_marked', ev.revolutions_class_marked, '', 'Rounded revolutions class over marked interval.', 0));
  lines.push(_numRow('underrotation_marked', ev.underrotation_marked, 'rev', 'Under/over rotation relative to the rounded class (marked interval).', 3));
  lines.push(_boolRow('underrot_flag_marked', ev.underrot_flag_marked, 'Under-rotation flag computed on marked interval (prototype).'));
  lines.push(_numRow('rotation_phase_marked', ev.rotation_phase_marked, '', 'Peak gyro timing fraction within the marked flight interval.', 3));

  lines.push('<div style="margin-top:8px;"><strong>Pose (video) metrics</strong></div>');
  lines.push(_numRow('flight_time_pose', ev.flight_time_pose, 's', 'Pose pipeline flight time (currently based on marked clip times).', 3));
  lines.push(_numRow('height_pose', ev.height_pose, 'm', 'Pose pipeline height estimate (currently from marked flight time).', 3));
  lines.push(_numRow('revolutions_pose', ev.revolutions_pose, 'rev', 'Pose pipeline rotation estimate (currently a 2D shoulder-line proxy).', 3));
  if (ev.pose_meta && typeof ev.pose_meta === 'object') {
    try {
      lines.push(_row('pose_meta', '<span class="small">stored</span>', 'Extra debug info for pose run (method, fps/stride, errors if any).'));
    } catch (e) {}
  }

  lines.push(sid ? _row('session_id', String(sid), 'Session folder that contains the source video.') : '<div class="small">No session linked (video playback unavailable)</div>');
  detail.innerHTML = lines.filter(Boolean).join('');

  if (jumpNameInput) jumpNameInput.value = ev.name || '';
  if (jumpNote) jumpNote.value = ev.note || '';
}

function clearSelectionAndPlots() {
  selectedIndex = -1;
  selectionAnchorIndex = -1;
  selectedJump = null;
  selectedJumpIds.clear();
  updateSelectionHighlight();
  renderSelectedDetail();
  // Clear plot series
  accX.length = 0; accY.length = 0; accZ.length = 0;
  gyroX.length = 0; gyroY.length = 0; gyroZ.length = 0;
  magX.length = 0; magY.length = 0; magZ.length = 0;
  _plotT0 = null;
  _plotT1 = null;
  redrawPlots(null, null, null, null, null, null);
}

function _collectIdxsInRange(arr, a, b) {
  const out = [];
  if (!Array.isArray(arr) || !arr.length) return out;
  for (let i = 0; i < arr.length; i++) {
    const tt = arr[i] && arr[i].t;
    if (typeof tt !== 'number') continue;
    if (tt < a || tt > b) continue;
    out.push(i);
  }
  return out;
}

function _downsampleIdxsPeakPreserving(arr, idxsIn, maxPoints) {
  // Preserve short spikes by selecting extrema per bucket (acc_mag, gyro_mag) + endpoints.
  if (!Array.isArray(idxsIn) || !idxsIn.length) return [];
  const N = idxsIn.length;
  const M = Math.max(10, Number(maxPoints || 150) || 150);
  if (N <= M) return idxsIn.slice();

  const bucketCount = Math.max(1, Math.floor(M / 2));
  const bucketSize = Math.max(1, Math.floor(N / bucketCount));
  const picked = new Set();

  function accMagAt(i) {
    try {
      const s = arr[i];
      const a = Array.isArray(s.acc) ? s.acc : [];
      const ax = Number(a[0] || 0), ay = Number(a[1] || 0), az = Number(a[2] || 0);
      return Math.sqrt(ax*ax + ay*ay + az*az);
    } catch (e) { return 0; }
  }
  function gyroMagAt(i) {
    try {
      const s = arr[i];
      const g = Array.isArray(s.gyro) ? s.gyro : [];
      const gx = Number(g[0] || 0), gy = Number(g[1] || 0), gz = Number(g[2] || 0);
      return Math.sqrt(gx*gx + gy*gy + gz*gz);
    } catch (e) { return 0; }
  }

  for (let b0 = 0; b0 < N; b0 += bucketSize) {
    const b1 = Math.min(N, b0 + bucketSize);
    const first = idxsIn[b0];
    const last = idxsIn[b1 - 1];
    picked.add(first);
    picked.add(last);

    let bestAccI = first, bestAccV = -1;
    let bestGyroI = first, bestGyroV = -1;
    for (let k = b0; k < b1; k++) {
      const i = idxsIn[k];
      const aV = accMagAt(i);
      if (aV > bestAccV) { bestAccV = aV; bestAccI = i; }
      const gV = gyroMagAt(i);
      if (gV > bestGyroV) { bestGyroV = gV; bestGyroI = i; }
    }
    picked.add(bestAccI);
    picked.add(bestGyroI);
  }

  const out = Array.from(picked).sort((a, b) => a - b);
  if (out.length > M) {
    const keep = [];
    const stride = Math.max(1, Math.floor(out.length / M));
    for (let i = 0; i < out.length; i += stride) keep.push(out[i]);
    if (keep[keep.length - 1] !== out[out.length - 1]) keep.push(out[out.length - 1]);
    return keep;
  }
  return out;
}

function _prepareFullClipPlot() {
  // Build a *fixed* plot covering the whole clip/session range, and later just move a cursor.
  if (!selectedJump || !Array.isArray(selectedJump.imu_samples)) return;
  const imu = selectedJump.imu_samples;

  // Determine plot range.
  let t0 = null, t1 = null;

  // IMPORTANT: Plot range must reflect IMU availability, not clip length.
  // Clips can be ~6s while IMU window is exactly 5s (t_peak-2 .. t_peak+3).
  // Prefer IMU sample timestamps first, then fall back to t_start/t_end.
  try {
    if (imu.length && typeof imu[0].t === 'number') t0 = imu[0].t;
    if (imu.length && typeof imu[imu.length - 1].t === 'number') t1 = imu[imu.length - 1].t;
  } catch (e) {}
  if (t0 == null && typeof selectedJump.t_start === 'number') t0 = selectedJump.t_start;
  if (t1 == null && typeof selectedJump.t_end === 'number') t1 = selectedJump.t_end;
  if (t0 == null && typeof selectedJump.t_peak === 'number') t0 = selectedJump.t_peak - 2.0;
  if (t1 == null && typeof selectedJump.t_peak === 'number') t1 = selectedJump.t_peak + 3.0;

  // Ensure ordering if imu_samples are unsorted for any reason.
  try {
    if (typeof t0 === 'number' && typeof t1 === 'number' && isFinite(t0) && isFinite(t1) && t1 <= t0 && imu.length > 1) {
      let mn = Infinity, mx = -Infinity;
      for (const s of imu) {
        const tt = s && s.t;
        if (typeof tt !== 'number' || !isFinite(tt)) continue;
        if (tt < mn) mn = tt;
        if (tt > mx) mx = tt;
      }
      if (isFinite(mn) && isFinite(mx) && mx > mn) {
        t0 = mn;
        t1 = mx;
      }
    }
  } catch (e) {}
  if (!(typeof t0 === 'number' && isFinite(t0) && typeof t1 === 'number' && isFinite(t1) && t1 > t0)) {
    _plotT0 = null; _plotT1 = null;
    return;
  }
  _plotT0 = t0;
  _plotT1 = t1;

  // Collect indices in range.
  const idxs = _collectIdxsInRange(imu, t0, t1);
  // Reset plot arrays.
  accX.length = accY.length = accZ.length = 0;
  gyroX.length = gyroY.length = gyroZ.length = 0;
  magX.length = magY.length = magZ.length = 0;

  if (!idxs.length) {
    redrawPlots(null, null, (t1 - t0), t0, t1, null);
    return;
  }

  // Plot ALL points for jump windows (expected ~5s @ 104Hz ≈ 520 samples).
  // This avoids any ambiguity from downsampling and is fast enough for the browser.
  for (let k = 0; k < idxs.length; k++) {
    const s = imu[idxs[k]];
    const a = s.acc || [];
    const g = s.gyro || [];
    const m = s.mag || [];
    accX.push(a[0] ?? 0); accY.push(a[1] ?? 0); accZ.push(a[2] ?? 0);
    gyroX.push(g[0] ?? 0); gyroY.push(g[1] ?? 0); gyroZ.push(g[2] ?? 0);
    magX.push(m[0] ?? 0); magY.push(m[1] ?? 0); magZ.push(m[2] ?? 0);
  }

  // Draw once (cursor will be added on updates).
  redrawPlots(null, 0.0, (t1 - t0), t0, t1, t0);
}

async function loadJumpList() {
  if (!jumpListEl) {
    console.warn('vLoad jumps: jumpListEl not found');
    return;
  }
  addLog('loadJumpList started');
  if (dbStatus) dbStatus.textContent = 'Loading...';
  try {
    let list = [];
    const serverRendered = jumpListEl.children.length > 0 && jumpListEl.children[0].hasAttribute && jumpListEl.children[0].hasAttribute('data-jump-id');
    addLog('serverRendered=' + serverRendered + ' children=' + jumpListEl.children.length);
    if (serverRendered) {
      // Build list from server-rendered <li data-jump-id="..." data-event-id="..." data-name="..." data-t-peak="...">
      for (let i = 0; i < jumpListEl.children.length; i++) {
        const li = jumpListEl.children[i];
        const jumpId = parseInt(li.getAttribute('data-jump-id'), 10);
        const eventIdRaw = li.getAttribute('data-event-id');
        const eventId = eventIdRaw !== null && eventIdRaw !== '' ? parseInt(eventIdRaw, 10) : null;
        const name = li.getAttribute('data-name') || '';
        const tPeakRaw = li.getAttribute('data-t-peak');
        const tPeak = tPeakRaw !== null && tPeakRaw !== '' ? parseFloat(tPeakRaw) : null;
        list.push({ jump_id: jumpId, event_id: eventId, name: name, t_peak: tPeak });
      }
    } else if (typeof window.__PRELOADED_JUMPSS__ !== 'undefined' && Array.isArray(window.__PRELOADED_JUMPSS__)) {
      list = window.__PRELOADED_JUMPSS__;
    } else {
      const resp = await fetch('/db/jumps?limit=200');
      if (!resp.ok) {
        if (dbStatus) dbStatus.textContent = `Failed (${resp.status})`;
        addLog('DB list failed: ' + resp.status);
        return;
      }
      const data = await resp.json();
      list = data.jumps || [];
    }
    jumps.length = 0;
    if (!serverRendered) {
      jumpListEl.innerHTML = '';
    }
    for (let i = 0; i < list.length && i < maxJumpItems; i++) {
      const ev = list[i];
      jumps.push(ev);
      let li, label;
      if (serverRendered && i < jumpListEl.children.length) {
        li = jumpListEl.children[i];
        label = li.querySelector('label');
      } else {
        const name = ev.name || (typeof ev.event_id === 'number' ? ('Jump ' + ev.event_id) : ('Jump ' + (i + 1)));
        const timeLabel = formatTimeFromEpoch(ev.t_peak);
        li = document.createElement('li');
        label = document.createElement('label');
        label.textContent = name + (timeLabel ? (' (' + timeLabel + ')') : '');
        li.appendChild(label);
        jumpListEl.appendChild(li);
      }
      // Click: plain = select only this; Shift = range; Ctrl/Cmd = toggle multi-select
      li.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        handleJumpListClick(e, i);
      });
    }
    if (dbStatus) dbStatus.textContent = `Loaded ${jumps.length}`;
    addLog(`Loaded ${jumps.length} jumps from DB.`);
    addLog('Click handlers attached to ' + jumps.length + ' rows');
    // reset selection if list changed
    clearSelectionAndPlots();
  } catch (e) {
    if (dbStatus) dbStatus.textContent = 'Error';
    addLog('DB load error: ' + e);
  }
}

function _findJumpIndexByJumpId(jumpId) {
  const jid = Number(jumpId);
  if (!isFinite(jid)) return -1;
  for (let i = 0; i < jumps.length; i++) {
    const r = jumps[i];
    if (r && Number(r.jump_id) === jid) return i;
  }
  return -1;
}

function _renderJumpListItemText(row) {
  if (!row) return '';
  const name = row.name || (typeof row.event_id === 'number' ? ('Jump ' + row.event_id) : 'Jump');
  const timeLabel = formatTimeFromEpoch(row.t_peak);
  return name + (timeLabel ? (' (' + timeLabel + ')') : '');
}

function _updateJumpListItemByIndex(i) {
  try {
    if (!jumpListEl) return;
    if (typeof i !== 'number' || i < 0) return;
    const li = jumpListEl.children && jumpListEl.children[i];
    const row = jumps[i];
    if (li && row) li.textContent = _renderJumpListItemText(row);
  } catch (e) {}
}

async function ensureVideoStartHost(sessionId) {
  videoStartHost = null;
  sessionFrames = null;
  try {
    const fr = await fetch('/sessions/' + sessionId + '/frames');
    if (!fr.ok) return;
    const fdata = await fr.json();
    const frames = fdata.frames || [];
    if (frames.length && typeof frames[0].t_host === 'number') {
      videoStartHost = frames[0].t_host;
      sessionFrames = frames;
    }
  } catch (e) {
    // ignore
  }
}

function handleJumpListClick(e, i) {
  const row = jumps[i];
  const jumpId = row && typeof row.jump_id === 'number' ? row.jump_id : null;
  if (e.shiftKey) {
    const anchor = selectionAnchorIndex < 0 ? i : selectionAnchorIndex;
    const low = Math.min(anchor, i);
    const high = Math.max(anchor, i);
    selectedJumpIds.clear();
    for (let k = low; k <= high; k++) {
      const r = jumps[k];
      if (r && typeof r.jump_id === 'number') selectedJumpIds.add(r.jump_id);
    }
    selectedIndex = i;
    updateSelectionHighlight();
    selectJump(i);
  } else if (e.ctrlKey || e.metaKey) {
    if (jumpId !== null) {
      if (selectedJumpIds.has(jumpId)) selectedJumpIds.delete(jumpId);
      else selectedJumpIds.add(jumpId);
    }
    selectionAnchorIndex = i;
    selectedIndex = i;
    updateSelectionHighlight();
    selectJump(i);
  } else {
    selectedJumpIds.clear();
    if (jumpId !== null) selectedJumpIds.add(jumpId);
    selectionAnchorIndex = i;
    selectedIndex = i;
    updateSelectionHighlight();
    selectJump(i);
  }
}

async function selectJump(index) {
  selectedIndex = index;
  const row = jumps[index];
  addLog('selectJump(' + index + ') row=' + (row ? 'jump_id=' + row.jump_id : 'null'));
  updateSelectionHighlight();
  if (!row || typeof row.jump_id !== 'number') {
    addLog('selectJump: no row or invalid jump_id, aborting');
    return;
  }
  addLog('Loading jump jump_id=' + row.jump_id + ' (event_id=' + row.event_id + ')...');
  try {
    const url = '/db/jumps/' + row.jump_id;
    const resp = await fetch(url);
    if (!resp.ok) {
      addLog('Jump detail fetch failed: ' + resp.status + ' ' + url);
      return;
    }
    selectedJump = await resp.json();
    addLog('Jump detail loaded jump_id=' + (selectedJump && selectedJump.jump_id));
    renderSelectedDetail();
    _prepareFullClipPlot();

    // Load per-jump clip if available; fall back to session video.
    const sid = selectedJump.session_id;
    const clipPath = selectedJump.video_path;
    if (clipPath && videoPlayer) {
      if (playerStatus) playerStatus.textContent = 'Loading clip...';
      videoPlayer.src = '/files?path=' + encodeURIComponent(clipPath);
      videoPlayer.load();
      // For clip playback, host alignment is "clip-local"; we can still update plots
      // by mapping video time to host time using t_peak heuristic (good enough for MVP).
      videoStartHost = null;
      if (playerStatus) playerStatus.textContent = 'Ready (clip)';
    } else if (sid && videoPlayer) {
      if (playerStatus) playerStatus.textContent = 'Loading video...';
      videoPlayer.src = '/sessions/' + sid + '/video';
      videoPlayer.load();
      await ensureVideoStartHost(sid);
      if (playerStatus) {
        playerStatus.textContent = videoStartHost != null
          ? ('Ready (session ' + sid + ')')
          : ('Ready (session ' + sid + ', frame timing not available yet)');
      }
    } else {
      if (videoPlayer) {
        videoPlayer.pause();
        videoPlayer.removeAttribute('src');
        videoPlayer.load();
      }
      if (playerStatus) playerStatus.textContent = 'No session/video linked';
    }

    // Initialize cursor around t_peak (even before playing)
    if (typeof selectedJump.t_peak === 'number') updatePlotsForHostTime(selectedJump.t_peak);
  } catch (e) {
    addLog('Select jump error: ' + e);
    if (e && e.stack) addLog(e.stack);
  }
}

function stopPlaybackLoop() {
  if (playbackRAF) {
    cancelAnimationFrame(playbackRAF);
    playbackRAF = null;
  }
  if (playbackVfcHandle != null && videoPlayer && typeof videoPlayer.cancelVideoFrameCallback === 'function') {
    try { videoPlayer.cancelVideoFrameCallback(playbackVfcHandle); } catch (e) { /* ignore */ }
  }
  playbackVfcHandle = null;
}

function _smoothNow() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function') ? performance.now() : Date.now();
}

function _stopSmoothCursor() {
  if (_smoothCursorRAF) {
    cancelAnimationFrame(_smoothCursorRAF);
    _smoothCursorRAF = null;
  }
  _smoothAnchorWall = null;
  _smoothAnchorHost = null;
}

function _anchorSmoothCursorAtHost(thHost) {
  const th = (typeof thHost === 'number' && isFinite(thHost)) ? Number(thHost) : null;
  if (th == null) return;
  _smoothAnchorHost = th;
  _smoothAnchorWall = _smoothNow();
}

function _smoothCursorLoop() {
  if (!videoPlayer) { _stopSmoothCursor(); return; }

  // If video ended and unified playback expects the plot to continue, fall into the host-tail loop.
  if (videoPlayer.ended) {
    _stopSmoothCursor();
    const bounds = _boundsCache || _computeTimelineBounds();
    if (_unifiedActive && bounds && bounds.globalEnd != null && bounds.videoEnd != null && bounds.globalEnd > bounds.videoEnd + 1e-3) {
      _startHostTailFrom(bounds.videoEnd);
    }
    return;
  }
  if (videoPlayer.paused) { _stopSmoothCursor(); return; }

  const bounds = _boundsCache || _computeTimelineBounds();
  const now = _smoothNow();
  const rate = Number(videoPlayer.playbackRate || 1.0) || 1.0;

  // Initialize anchor from currentTime if needed.
  if (_smoothAnchorWall == null || _smoothAnchorHost == null) {
    const th0 = hostTimeFromVideoTime(videoPlayer.currentTime);
    if (typeof th0 === 'number' && isFinite(th0)) _anchorSmoothCursorAtHost(th0);
  }
  if (_smoothAnchorWall == null || _smoothAnchorHost == null) {
    _stopSmoothCursor();
    return;
  }

  let t = _smoothAnchorHost + (Math.max(0, (now - _smoothAnchorWall)) / 1000.0) * rate;

  // If drift grows too large, snap to actual video time and re-anchor.
  const thFrame = hostTimeFromVideoTime(videoPlayer.currentTime);
  if (typeof thFrame === 'number' && isFinite(thFrame) && Math.abs(t - thFrame) > 0.25) {
    _anchorSmoothCursorAtHost(thFrame);
    t = thFrame;
  }

  // Enforce unified stop bound when globalEnd is reached (even if the next video frame hasn't arrived yet).
  if (bounds && bounds.globalEnd != null && isFinite(bounds.globalEnd) && t >= bounds.globalEnd) {
    try { videoPlayer.pause(); } catch (e) {}
    _setCursorHost(bounds.globalEnd, { noSeekVideo: true });
    _stopSmoothCursor();
    stopPlaybackLoop();
    _unifiedActive = false;
    return;
  }

  // Update cursor smoothly without seeking video (video is master clock).
  _setCursorHost(t, { noSeekVideo: true });
  _smoothCursorRAF = requestAnimationFrame(_smoothCursorLoop);
}

function _clamp(x, a, b) {
  const xx = Number(x);
  const aa = Number(a);
  const bb = Number(b);
  if (!isFinite(xx) || !isFinite(aa) || !isFinite(bb) || bb < aa) return xx;
  return Math.max(aa, Math.min(bb, xx));
}

function _computeTimelineBounds() {
  // Prefer deterministic bounds from DB-provided timestamps.
  // Plot bounds (IMU availability)
  const plotStart = (typeof _plotT0 === 'number' && isFinite(_plotT0)) ? Number(_plotT0) : null;
  const plotEnd = (typeof _plotT1 === 'number' && isFinite(_plotT1)) ? Number(_plotT1) : null;

  // Video bounds (frame host timestamps if available)
  let videoStart = null, videoEnd = null;
  try {
    const kind = _playbackKind();
    if (kind === 'session' && Array.isArray(sessionFrames) && sessionFrames.length) {
      const a = sessionFrames[0];
      const b = sessionFrames[sessionFrames.length - 1];
      if (a && typeof a.t_host === 'number') videoStart = Number(a.t_host);
      if (b && typeof b.t_host === 'number') videoEnd = Number(b.t_host);
    } else if (selectedJump && Array.isArray(selectedJump.frames) && selectedJump.frames.length) {
      const a = selectedJump.frames[0];
      const b = selectedJump.frames[selectedJump.frames.length - 1];
      if (a && typeof a.t_host === 'number') videoStart = Number(a.t_host);
      if (b && typeof b.t_host === 'number') videoEnd = Number(b.t_host);
    }
  } catch (e) {}

  // Fallback: if we have an anchor + duration, approximate (best-effort)
  try {
    if (videoStart == null && typeof hostTimeFromVideoTime === 'function') {
      const th0 = hostTimeFromVideoTime(0);
      if (typeof th0 === 'number' && isFinite(th0)) videoStart = Number(th0);
    }
    if (videoEnd == null && videoStart != null && videoPlayer && typeof videoPlayer.duration === 'number' && isFinite(videoPlayer.duration) && videoPlayer.duration > 0) {
      videoEnd = Number(videoStart) + Number(videoPlayer.duration);
    }
  } catch (e) {}

  const starts = [];
  if (plotStart != null) starts.push(plotStart);
  if (videoStart != null) starts.push(videoStart);
  const ends = [];
  if (plotEnd != null) ends.push(plotEnd);
  if (videoEnd != null) ends.push(videoEnd);

  const globalStart = starts.length ? Math.min(...starts) : null;
  const globalEnd = ends.length ? Math.max(...ends) : null; // stop when last track ends (requested)

  const out = { plotStart, plotEnd, videoStart, videoEnd, globalStart, globalEnd };
  _boundsCache = out;
  return out;
}

function _plotCursorHostForTimeline(th, bounds) {
  // Plot should not "invent" time outside IMU data; clamp cursor to plot range if known.
  if (!bounds) bounds = _boundsCache || _computeTimelineBounds();
  if (bounds && bounds.plotStart != null && bounds.plotEnd != null) return _clamp(th, bounds.plotStart, bounds.plotEnd);
  return th;
}

function _seekVideoForHostTime(th, bounds) {
  if (!videoPlayer) return;
  if (!bounds) bounds = _boundsCache || _computeTimelineBounds();
  if (!bounds || bounds.videoStart == null || bounds.videoEnd == null) return;

  // Gate video: before start -> hold at t=0; after end -> hold at end.
  if (th < bounds.videoStart) {
    try { videoPlayer.pause(); } catch (e) {}
    try { videoPlayer.currentTime = 0; } catch (e) {}
    return;
  }
  if (th > bounds.videoEnd) {
    try { videoPlayer.pause(); } catch (e) {}
    try {
      if (typeof videoPlayer.duration === 'number' && isFinite(videoPlayer.duration) && videoPlayer.duration > 0) {
        videoPlayer.currentTime = Math.max(0, videoPlayer.duration - (1 / 60));
      } else {
        const tv = videoTimeFromHostTime(bounds.videoEnd);
        if (typeof tv === 'number' && isFinite(tv)) videoPlayer.currentTime = Math.max(0, tv);
      }
    } catch (e) {}
    return;
  }

  // In range: seek by host time mapping.
  const tv = videoTimeFromHostTime(th);
  if (typeof tv === 'number' && isFinite(tv)) {
    try { videoPlayer.currentTime = Math.max(0, tv); } catch (e) {}
  }
}

function _setCursorHost(th, opts) {
  const o = opts || {};
  const bounds = _boundsCache || _computeTimelineBounds();
  const t = (typeof th === 'number' && isFinite(th)) ? Number(th) : null;
  if (t == null) return;
  _cursorHost = t;
  // Update plot cursor (clamped to IMU range).
  const tPlot = _plotCursorHostForTimeline(t, bounds);
  updatePlotsForHostTime(tPlot);
  // Seek video unless caller says not to (e.g., when video itself is driving time).
  if (!o.noSeekVideo) _seekVideoForHostTime(t, bounds);
  try {
    if (playerStatus) playerStatus.textContent = `host=${formatTimeFromEpoch(t)}`;
  } catch (e) {}
}

function updatePlotsForHostTime(tHost) {
  if (!selectedJump || !Array.isArray(selectedJump.imu_samples)) return;
  // If we have a precomputed full-clip plot range, just move the cursor.
  if (typeof _plotT0 === 'number' && typeof _plotT1 === 'number' && isFinite(_plotT0) && isFinite(_plotT1) && _plotT1 > _plotT0) {
    const frac = (tHost - _plotT0) / (_plotT1 - _plotT0);
    redrawPlots(null, frac, (_plotT1 - _plotT0), _plotT0, _plotT1, tHost);
    // Show wall-clock time for the current cursor so sync issues are obvious.
    try {
      if (playerStatus) {
        playerStatus.textContent = `host=${formatTimeFromEpoch(tHost)}`;
      }
    } catch (e) {}
    return;
  }

  const src = selectedJump.imu_samples;
  const windowSec = 3.0;
  const t0 = tHost - windowSec;
  const t1 = tHost;

  accX.length = accY.length = accZ.length = 0;
  gyroX.length = gyroY.length = gyroZ.length = 0;
  magX.length = magY.length = magZ.length = 0;

  // Plot ALL points in the 3s window (expected ~312 samples @ 104Hz).
  const idxs = _collectIdxsInRange(src, t0, t1);
  for (const ii of idxs) {
    const s = src[ii];
    const a = s.acc || [];
    const g = s.gyro || [];
    const m = s.mag || [];
    pushLimited(accX, a[0] ?? 0); pushLimited(accY, a[1] ?? 0); pushLimited(accZ, a[2] ?? 0);
    pushLimited(gyroX, g[0] ?? 0); pushLimited(gyroY, g[1] ?? 0); pushLimited(gyroZ, g[2] ?? 0);
    pushLimited(magX, m[0] ?? 0); pushLimited(magY, m[1] ?? 0); pushLimited(magZ, m[2] ?? 0);
  }
  redrawPlots(50, null, windowSec, t0, t1, tHost);
  try {
    if (playerStatus) {
      playerStatus.textContent = `host=${formatTimeFromEpoch(tHost)}`;
    }
  } catch (e) {}
}

function _playbackKind() {
  // The <video> element always speaks "media time" (seconds from start of current file).
  // We still need to know which file is loaded to do correct host<->video mapping.
  const src = (videoPlayer && (videoPlayer.currentSrc || videoPlayer.src)) ? String(videoPlayer.currentSrc || videoPlayer.src) : '';
  if (src.includes('/sessions/') && src.includes('/video')) return 'session';
  // Jump clips are served via /files?path=...
  return 'clip';
}

function _binarySearchFloor(arr, value, getter) {
  if (!Array.isArray(arr) || !arr.length) return 0;
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi + 1) / 2);
    const mv = Number(getter(arr[mid]) || 0);
    if (mv <= value) lo = mid; else hi = mid - 1;
  }
  return lo;
}

function _clipStartHostEstimate() {
  // Estimate host epoch time that corresponds to clip t_video=0.
  // This allows continuous host<->video mapping even if jump_frames only cover
  // a subset of frames (e.g., ffprobe timestamps missing for some frames).
  try {
    if (selectedJump && Array.isArray(selectedJump.frames) && selectedJump.frames.length) {
      const f0 = selectedJump.frames[0];
      const th0 = (f0 && typeof f0.t_host === 'number') ? Number(f0.t_host) : null;
      const tv0 = (f0 && typeof f0.t_video === 'number') ? Number(f0.t_video) : 0.0;
      if (typeof th0 === 'number' && isFinite(th0) && isFinite(tv0)) return th0 - tv0;
    }
  } catch (e) {}
  return null;
}

// Convert current file's video time (seconds-from-file-start) -> host epoch seconds.
function hostTimeFromVideoTime(videoTimeSec) {
  const kind = _playbackKind();
  const tv = Number(videoTimeSec || 0);

  if (kind === 'session') {
    // Prefer exact session frame mapping if available.
    if (videoStartHost != null && Array.isArray(sessionFrames) && sessionFrames.length) {
      const idx = Math.max(1, Math.floor(tv * sessionFps) + 1); // frames.csv frame_idx starts at 1
      const arrIdx = Math.min(sessionFrames.length - 1, Math.max(0, idx - 1));
      const f = sessionFrames[arrIdx];
      if (f && typeof f.t_host === 'number') return f.t_host;
    }
    if (videoStartHost != null) return Number(videoStartHost) + tv;
  }

  // Clip playback: prefer per-jump clip frames (clip-relative t_video ~ 0..6).
  const clipStartHost = _clipStartHostEstimate();
  if (clipStartHost != null) return Number(clipStartHost) + tv;
  if (selectedJump && Array.isArray(selectedJump.frames) && selectedJump.frames.length) {
    const frames = selectedJump.frames;
    const i = _binarySearchFloor(frames, tv, (x) => x.t_video);
    const f = frames[i];
    if (f && typeof f.t_host === 'number') return f.t_host;
  }

  // Fallback: if we have a session anchor, approximate host time linearly.
  if (videoStartHost != null) return Number(videoStartHost) + tv;

  // Last resort heuristic
  if (selectedJump && typeof selectedJump.t_peak === 'number') return selectedJump.t_peak + (tv - 1.0);
  return null;
}

// Convert host epoch seconds -> current file's video time (seconds-from-file-start).
function videoTimeFromHostTime(hostSec) {
  const kind = _playbackKind();
  const th = Number(hostSec || 0);
  if (!isFinite(th)) return null;

  if (kind === 'session') {
    if (videoStartHost != null && Array.isArray(sessionFrames) && sessionFrames.length) {
      const i = _binarySearchFloor(sessionFrames, th, (x) => x.t_host);
      const f = sessionFrames[i];
      // sessionFrames are 1-based frame_idx (frames.csv); convert to seconds.
      if (f && typeof f.frame_idx === 'number') return Math.max(0, (Number(f.frame_idx) - 1) / (Number(sessionFps || 30) || 30));
      if (f && typeof f.t_host === 'number' && videoStartHost != null) return Math.max(0, Number(f.t_host) - Number(videoStartHost));
    }
    if (videoStartHost != null) return Math.max(0, th - Number(videoStartHost));
    return null;
  }

  // Clip playback: prefer exact mapping via per-frame host timestamps.
  const clipStartHost = _clipStartHostEstimate();
  if (clipStartHost != null) return Math.max(0, th - Number(clipStartHost));
  if (selectedJump && Array.isArray(selectedJump.frames) && selectedJump.frames.length) {
    const frames = selectedJump.frames;
    const i = _binarySearchFloor(frames, th, (x) => x.t_host);
    const f = frames[i];
    if (f && typeof f.t_video === 'number') return Math.max(0, Number(f.t_video));
    // If clip frames exist but no t_video (shouldn't happen), fall back to anchor.
    if (f && typeof f.t_host === 'number') return Math.max(0, th - Number(f.t_host));
  }

  // Fallback: approximate using session anchor if available.
  if (videoStartHost != null) return Math.max(0, th - Number(videoStartHost));
  return null;
}

// Host-time first: get nearest frame index for the ACTIVE timeline.
function _nearestFrameIndexForVideoTime(videoTimeSec) {
  const th = hostTimeFromVideoTime(videoTimeSec);
  if (typeof th !== 'number' || !isFinite(th)) {
    const fps = Number(sessionFps || 30) || 30;
    return Math.max(0, Math.floor((Number(videoTimeSec || 0) * fps)));
  }
  const kind = _playbackKind();
  if (kind === 'session' && Array.isArray(sessionFrames) && sessionFrames.length) {
    return _binarySearchFloor(sessionFrames, th, (x) => x.t_host);
  }
  if (selectedJump && Array.isArray(selectedJump.frames) && selectedJump.frames.length) {
    return _binarySearchFloor(selectedJump.frames, th, (x) => x.t_host);
  }
  const fps = Number(sessionFps || 30) || 30;
  return Math.max(0, Math.floor((Number(videoTimeSec || 0) * fps)));
}

function _seekToFrameIndex(idx) {
  if (!videoPlayer) return;
  const kind = _playbackKind();
  const iRaw = Number(idx || 0);
  const fps = Number(sessionFps || 30) || 30;

  if (kind === 'session' && Array.isArray(sessionFrames) && sessionFrames.length) {
    const i = Math.max(0, Math.min(sessionFrames.length - 1, iRaw));
    const f = sessionFrames[i];
    const th = (f && typeof f.t_host === 'number') ? Number(f.t_host) : null;
    const tv = (f && typeof f.frame_idx === 'number') ? Math.max(0, (Number(f.frame_idx) - 1) / fps) : (th != null ? videoTimeFromHostTime(th) : null);
    if (typeof tv === 'number' && isFinite(tv)) videoPlayer.currentTime = Math.max(0, tv);
    if (typeof th === 'number') updatePlotsForHostTime(th);
    if (playerStatus && typeof th === 'number') playerStatus.textContent = `Frame ${i + 1}/${sessionFrames.length}  host=${formatTimeFromEpoch(th)}`;
    return;
  }

  if (selectedJump && Array.isArray(selectedJump.frames) && selectedJump.frames.length) {
    const frames = selectedJump.frames;
    const i = Math.max(0, Math.min(frames.length - 1, iRaw));
    const f = frames[i];
    // Seek by host time (host-time only approach) -> convert to video time for current file.
    const th = (f && typeof f.t_host === 'number') ? Number(f.t_host) : null;
    const tv = (th != null) ? videoTimeFromHostTime(th) : (typeof f.t_video === 'number' ? Number(f.t_video) : null);
    if (typeof tv === 'number' && isFinite(tv)) videoPlayer.currentTime = Math.max(0, tv);
    if (typeof th === 'number') updatePlotsForHostTime(th);
    if (playerStatus && typeof th === 'number') playerStatus.textContent = `Frame ${i + 1}/${frames.length}  host=${formatTimeFromEpoch(th)}`;
    return;
  }

  // Fallback: approximate by FPS
  const tv = Math.max(0, (iRaw / fps));
  videoPlayer.currentTime = tv;
  const th = hostTimeFromVideoTime(tv);
  if (typeof th === 'number') updatePlotsForHostTime(th);
  if (playerStatus) playerStatus.textContent = `Frame ~${iRaw}`;
}

// Robust single-frame stepping: decode the next frame in sequence (doesn't rely on random-access seek).
// This avoids getting "stuck" when MP4 seeks land on the same keyframe.
async function _stepOneFrameForward() {
  if (!videoPlayer) return false;
  // Ensure paused before stepping
  try { videoPlayer.pause(); } catch (e) {}

  // requestVideoFrameCallback is the most reliable way to advance by a single decoded frame
  if (typeof videoPlayer.requestVideoFrameCallback === 'function') {
    return await new Promise(async (resolve) => {
      let done = false;
      const onFrame = (_now, meta) => {
        if (done) return;
        done = true;
        try { videoPlayer.pause(); } catch (e) {}
        try { stopPlaybackLoop(); } catch (e) {}
        const tv = (meta && typeof meta.mediaTime === 'number') ? meta.mediaTime : videoPlayer.currentTime;
        const th = hostTimeFromVideoTime(tv);
        if (typeof th === 'number') updatePlotsForHostTime(th);
        if (playerStatus && typeof th === 'number') playerStatus.textContent = `host=${formatTimeFromEpoch(th)}`;
        resolve(true);
      };
      try { videoPlayer.requestVideoFrameCallback(onFrame); } catch (e) { /* ignore */ }
      try {
        const p = videoPlayer.play();
        if (p && typeof p.catch === 'function') p.catch(() => {});
      } catch (e) {
        resolve(false);
      }
      // Safety timeout in case the browser doesn't deliver a frame callback
      setTimeout(() => {
        if (done) return;
        done = true;
        try { videoPlayer.pause(); } catch (e) {}
        resolve(false);
      }, 700);
    });
  }

  // Fallback: tiny forward seek (less reliable with GOPs)
  try {
    const dt = 1.0 / 30.0;
    videoPlayer.currentTime = Math.max(0, (videoPlayer.currentTime || 0) + dt);
    const th = hostTimeFromVideoTime(videoPlayer.currentTime);
    if (typeof th === 'number') updatePlotsForHostTime(th);
    if (playerStatus && typeof th === 'number') playerStatus.textContent = `host=${formatTimeFromEpoch(th)}`;
    return true;
  } catch (e) {
    return false;
  }
}

async function stepFrame(delta) {
  if (!selectedJump || !videoPlayer) {
    addLog('Select a jump first.');
    return;
  }
  try {
    videoPlayer.pause();
  } catch (e) {}
  // Frame-stepping should be done at real-time speed to keep the decode behavior stable.
  try { videoPlayer.playbackRate = 1.0; } catch (e) {}
  _stopUnifiedPlayback();
  const d = Number(delta || 0);

  const bounds = _boundsCache || _computeTimelineBounds();
  if (!bounds || bounds.globalStart == null || bounds.globalEnd == null) {
    // Fallback to previous behavior if bounds are missing.
    const curIdx = _nearestFrameIndexForVideoTime(videoPlayer.currentTime);
    _seekToFrameIndex(curIdx + d);
    return;
  }

  // Determine current timeline host time (cursorHost -> video position -> fallback to t_peak)
  let thCur = (typeof _cursorHost === 'number' && isFinite(_cursorHost)) ? Number(_cursorHost) : null;
  if (thCur == null) {
    const thV = hostTimeFromVideoTime(videoPlayer.currentTime);
    if (typeof thV === 'number' && isFinite(thV)) thCur = Number(thV);
  }
  if (thCur == null && selectedJump && typeof selectedJump.t_peak === 'number') thCur = Number(selectedJump.t_peak);
  if (thCur == null) thCur = bounds.globalStart;

  // Estimate a frame duration (host-time). Prefer observed frame cadence if available.
  let dt = 1.0 / 30.0;
  try {
    const kind = _playbackKind();
    const frames = (kind === 'session') ? sessionFrames : (selectedJump ? selectedJump.frames : null);
    if (Array.isArray(frames) && frames.length >= 2) {
      const dth = Number(frames[1].t_host) - Number(frames[0].t_host);
      if (isFinite(dth) && dth > 1e-4 && dth < 0.2) dt = dth;
    }
  } catch (e) {}

  // If we're inside the video's host-time window, step to the next/prev actual frame host time.
  // If we're before videoStart (plot-only lead-in), step by dt and keep video parked at 0.
  let thNew = null;
  if (bounds.videoStart != null && bounds.videoEnd != null && thCur >= bounds.videoStart && thCur <= bounds.videoEnd) {
    try {
      const kind = _playbackKind();
      const frames = (kind === 'session') ? sessionFrames : (selectedJump ? selectedJump.frames : null);
      if (Array.isArray(frames) && frames.length) {
        const idx = _binarySearchFloor(frames, thCur, (x) => x.t_host);
        const idxNew = Math.max(0, Math.min(frames.length - 1, idx + d));
        const f = frames[idxNew];
        if (f && typeof f.t_host === 'number') thNew = Number(f.t_host);
      }
    } catch (e) {}
  }
  if (thNew == null) {
    thNew = thCur + (d * dt);
  }

  thNew = _clamp(thNew, bounds.globalStart, bounds.globalEnd);
  _setCursorHost(thNew);
}

function playbackLoop() {
  if (!videoPlayer) { stopPlaybackLoop(); return; }
  if (videoPlayer.ended) {
    stopPlaybackLoop();
    const bounds = _boundsCache || _computeTimelineBounds();
    // If unified playback is active and plot extends beyond video, continue advancing host cursor.
    if (_unifiedActive && bounds && bounds.globalEnd != null && bounds.videoEnd != null && bounds.globalEnd > bounds.videoEnd + 1e-3) {
      _startHostTailFrom(bounds.videoEnd);
    }
    return;
  }
  if (videoPlayer.paused) {
    stopPlaybackLoop();
    return;
  }
  const bounds = _boundsCache || _computeTimelineBounds();
  const tHost = hostTimeFromVideoTime(videoPlayer.currentTime);
  if (typeof tHost === 'number') {
    // Stop when last track ends (globalEnd = max(plotEnd, videoEnd)).
    if (bounds && bounds.globalEnd != null && tHost >= bounds.globalEnd) {
      try { videoPlayer.pause(); } catch (e) {}
      _setCursorHost(bounds.globalEnd, { noSeekVideo: true });
      stopPlaybackLoop();
      _unifiedActive = false;
      return;
    }
    _setCursorHost(tHost, { noSeekVideo: true });
  }
  playbackRAF = requestAnimationFrame(playbackLoop);
}

function startFrameSyncedLoop() {
  if (!videoPlayer) return;

  // Start smooth cursor interpolation and keep re-anchoring it on decoded frames.
  try {
    const th0 = hostTimeFromVideoTime(videoPlayer.currentTime);
    if (typeof th0 === 'number' && isFinite(th0)) _anchorSmoothCursorAtHost(th0);
  } catch (e) {}
  if (!_smoothCursorRAF) _smoothCursorRAF = requestAnimationFrame(_smoothCursorLoop);

  if (typeof videoPlayer.requestVideoFrameCallback === 'function') {
    const onFrame = (_now, meta) => {
      if (!videoPlayer) { stopPlaybackLoop(); return; }
      if (videoPlayer.ended) {
        stopPlaybackLoop();
        const bounds = _boundsCache || _computeTimelineBounds();
        if (_unifiedActive && bounds && bounds.globalEnd != null && bounds.videoEnd != null && bounds.globalEnd > bounds.videoEnd + 1e-3) {
          _startHostTailFrom(bounds.videoEnd);
        }
        return;
      }
      if (videoPlayer.paused) {
        stopPlaybackLoop();
        return;
      }
      const tHost = hostTimeFromVideoTime((meta && typeof meta.mediaTime === 'number') ? meta.mediaTime : videoPlayer.currentTime);
      if (typeof tHost === 'number') {
        const bounds = _boundsCache || _computeTimelineBounds();
        if (bounds && bounds.globalEnd != null && tHost >= bounds.globalEnd) {
          try { videoPlayer.pause(); } catch (e) {}
          _setCursorHost(bounds.globalEnd, { noSeekVideo: true });
          stopPlaybackLoop();
          _unifiedActive = false;
          return;
        }
        // Re-anchor smooth interpolation to this decoded frame timestamp.
        _anchorSmoothCursorAtHost(tHost);
        // Occasionally snap to the exact frame time to avoid visible drift.
        try {
          if (_cursorHost == null || Math.abs(Number(_cursorHost) - Number(tHost)) > 0.05) {
            _setCursorHost(tHost, { noSeekVideo: true });
          }
        } catch (e) {}
      }
      playbackVfcHandle = videoPlayer.requestVideoFrameCallback(onFrame);
    };
    playbackVfcHandle = videoPlayer.requestVideoFrameCallback(onFrame);
    return;
  }
  // No requestVideoFrameCallback: don't drive the cursor from video.currentTime (which can be choppy);
  // the smooth cursor loop above will keep it moving and snap periodically using currentTime.
  playbackRAF = requestAnimationFrame(playbackLoop);
}

function _cancelHostPhase() {
  // Cancel any host-driven animation phase without ending "unified playback" overall.
  stopPlaybackLoop();
  _timelineMode = null;
  _timelineWallT0 = null;
  _timelineHostT0 = null;
  _timelineVideoStarted = false;
  _stopSmoothCursor();
}

function _stopUnifiedPlayback() {
  _unifiedActive = false;
  _cancelHostPhase();
}

function _unifiedHostTailLoop() {
  // Continue host cursor after the video has ended (plot still advancing).
  const bounds = _boundsCache || _computeTimelineBounds();
  if (!bounds || bounds.globalEnd == null) {
    _stopUnifiedPlayback();
    return;
  }
  const now = (typeof performance !== 'undefined' && typeof performance.now === 'function') ? performance.now() : Date.now();
  if (_timelineWallT0 == null || _timelineHostT0 == null) {
    _timelineWallT0 = now;
    _timelineHostT0 = (typeof _cursorHost === 'number' && isFinite(_cursorHost)) ? Number(_cursorHost) : (bounds.videoEnd != null ? Number(bounds.videoEnd) : bounds.globalStart);
  }
  const dt = Math.max(0, (now - _timelineWallT0) / 1000.0);
  const tHost = _timelineHostT0 + (dt * (Number(_timelineRate || 1.0) || 1.0));
  const tClamped = _clamp(tHost, _timelineHostT0, bounds.globalEnd);
  _setCursorHost(tClamped, { noSeekVideo: true });
  if (tClamped >= bounds.globalEnd) {
    _stopUnifiedPlayback();
    return;
  }
  playbackRAF = requestAnimationFrame(_unifiedHostTailLoop);
}

function _startHostTailFrom(thStart) {
  const bounds = _boundsCache || _computeTimelineBounds();
  if (!bounds || bounds.globalEnd == null) return;
  _cancelHostPhase();
  _timelineMode = 'host_post_video';
  _timelineWallT0 = null;
  _timelineHostT0 = (typeof thStart === 'number' && isFinite(thStart)) ? Number(thStart) : ((typeof _cursorHost === 'number' && isFinite(_cursorHost)) ? Number(_cursorHost) : bounds.globalStart);
  // Ensure video is frozen at end.
  try { if (videoPlayer) videoPlayer.pause(); } catch (e) {}
  _setCursorHost(_timelineHostT0, { noSeekVideo: true });
  playbackRAF = requestAnimationFrame(_unifiedHostTailLoop);
}

function _unifiedHostLoop() {
  // Host-driven phase used only when plot starts before video start.
  const bounds = _boundsCache || _computeTimelineBounds();
  if (!bounds || bounds.globalStart == null || bounds.globalEnd == null) {
    _stopUnifiedPlayback();
    return;
  }
  const now = (typeof performance !== 'undefined' && typeof performance.now === 'function') ? performance.now() : Date.now();
  if (_timelineWallT0 == null || _timelineHostT0 == null) {
    _timelineWallT0 = now;
    _timelineHostT0 = bounds.globalStart;
  }
  const dt = Math.max(0, (now - _timelineWallT0) / 1000.0);
  const tHost = _timelineHostT0 + (dt * (Number(_timelineRate || 1.0) || 1.0));
  const tClamped = _clamp(tHost, bounds.globalStart, bounds.globalEnd);
  _setCursorHost(tClamped);

  // Start video only once timeline reaches videoStart; after that, use video as master clock.
  if (!_timelineVideoStarted && bounds.videoStart != null && tClamped >= bounds.videoStart) {
    _timelineVideoStarted = true;
    // Seek video to current host time and start playing, then switch to frame-synced loop.
    _seekVideoForHostTime(tClamped, bounds);
    try { videoPlayer.playbackRate = Number(_timelineRate || 1.0) || 1.0; } catch (e) {}
    try {
      const p = videoPlayer.play();
      if (p && typeof p.catch === 'function') p.catch(() => {});
    } catch (e) {}
    _cancelHostPhase();
    startFrameSyncedLoop();
    return;
  }

  if (tClamped >= bounds.globalEnd) {
    try { if (videoPlayer) videoPlayer.pause(); } catch (e) {}
    _stopUnifiedPlayback();
    return;
  }

  playbackRAF = requestAnimationFrame(_unifiedHostLoop);
}

async function _startUnifiedPlayback(rate) {
  if (!selectedJump || !videoPlayer) {
    addLog('Select a jump first.');
    return;
  }
  if (!selectedJump.session_id) {
    addLog('Selected jump has no session_id; cannot play video.');
    return;
  }

  _timelineRate = Number(rate || 1.0) || 1.0;
  _computeTimelineBounds();
  const bounds = _boundsCache;
  if (!bounds || bounds.globalStart == null || bounds.globalEnd == null) {
    addLog('Cannot compute playback bounds (missing plot/video timing).');
    return;
  }
  _unifiedActive = true;

  // Start from whichever begins earlier (as requested).
  const startHost = bounds.globalStart;
  _cancelHostPhase();
  try { videoPlayer.pause(); } catch (e) {}
  try { videoPlayer.playbackRate = Number(_timelineRate || 1.0) || 1.0; } catch (e) {}

  // Position both tracks at the start of the shared timeline.
  _setCursorHost(startHost);

  // Decide which one should begin first.
  // If video starts earliest (or same), let video be the master clock immediately.
  // If plot starts earlier, run host-clock until reaching videoStart, then start video and switch master.
  const plotStart = bounds.plotStart;
  const videoStart = bounds.videoStart;
  if (videoStart != null && (plotStart == null || videoStart <= plotStart)) {
    // Seek video to its start point and play; plot will clamp until its own start.
    _seekVideoForHostTime(startHost, bounds);
    try {
      await videoPlayer.play();
      startFrameSyncedLoop();
    } catch (e) {
      addLog('Video play failed: ' + e);
    }
    return;
  }

  // Plot leads: keep video parked until timeline reaches videoStart.
  _timelineMode = 'host_pre_video';
  _timelineWallT0 = null;
  _timelineHostT0 = startHost;
  _timelineVideoStarted = false;
  playbackRAF = requestAnimationFrame(_unifiedHostLoop);
}

async function playAtRate(rate) {
  await _startUnifiedPlayback(Number(rate || 1.0) || 1.0);
}

async function markVideo(which) {
  if (!selectedJump || typeof selectedJump.jump_id !== 'number' || !videoPlayer) {
    addLog('Select a jump first.');
    return;
  }
  const tv = Number(videoPlayer.currentTime || 0);
  let th = hostTimeFromVideoTime(tv);
  if (typeof th !== 'number' || !isFinite(th)) {
    // Fallback: use session anchor if available
    if (videoStartHost != null) th = Number(videoStartHost) + tv;
  }
  if (typeof th !== 'number' || !isFinite(th)) {
    addLog('Cannot compute host time for this video position (missing frame mapping).');
    return;
  }
  try {
    if (playerStatus) playerStatus.textContent = `Saving ${which}...`;
    const resp = await fetch('/db/jumps/' + selectedJump.jump_id + '/marks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ which, t_host: th, t_video: tv })
    });
    if (!resp.ok) {
      let msg = 'Mark failed (' + resp.status + ')';
      try {
        const err = await resp.json();
        if (err && err.detail) msg += ': ' + err.detail;
      } catch (e) {
        try { msg += ': ' + (await resp.text()); } catch (e2) {}
      }
      addLog(msg);
      if (playerStatus) playerStatus.textContent = `Mark failed (${resp.status})`;
      return;
    }
    const data = await resp.json();
    // Update local selectedJump so the detail panel updates immediately
    if (typeof data.t_takeoff_video === 'number') selectedJump.t_takeoff_video = data.t_takeoff_video;
    if (typeof data.t_takeoff_video_t === 'number') selectedJump.t_takeoff_video_t = data.t_takeoff_video_t;
    if (typeof data.t_landing_video === 'number') selectedJump.t_landing_video = data.t_landing_video;
    if (typeof data.t_landing_video_t === 'number') selectedJump.t_landing_video_t = data.t_landing_video_t;
    renderSelectedDetail();
    addLog(`Saved ${which}: host=${formatTimeFromEpoch(th)} (t_host=${th.toFixed(3)}), video_time=${tv.toFixed(3)}s`);
    if (playerStatus) playerStatus.textContent = `Saved ${which}`;
  } catch (e) {
    addLog('Mark error: ' + e);
    if (playerStatus) playerStatus.textContent = 'Mark error';
  }
}

async function recomputeMarked() {
  if (!selectedJump || typeof selectedJump.jump_id !== 'number') {
    addLog('Select a jump first.');
    return;
  }
  const jumpId = selectedJump.jump_id;
  try {
    if (playerStatus) playerStatus.textContent = 'Recomputing...';
    const resp = await fetch('/db/jumps/' + jumpId + '/recompute_marked_metrics', { method: 'POST' });
    if (!resp.ok) {
      let msg = 'Recompute failed (' + resp.status + ')';
      try {
        const err = await resp.json();
        if (err && err.detail) msg += ': ' + err.detail;
      } catch (e) {
        try { msg += ': ' + (await resp.text()); } catch (e2) {}
      }
      addLog(msg);
      if (playerStatus) playerStatus.textContent = `Recompute failed (${resp.status})`;
      return;
    }
    const data = await resp.json();
    // Update local fields so the detail panel updates immediately
    const keys = [
      'flight_time_marked','height_marked','rotation_phase_marked',
      'theta_z_rad_marked','revolutions_est_marked','revolutions_class_marked',
      'underrotation_marked','underrot_flag_marked','gz_bias_marked'
    ];
    keys.forEach(k => {
      if (data && Object.prototype.hasOwnProperty.call(data, k)) selectedJump[k] = data[k];
    });
    renderSelectedDetail();
    addLog('Recompute done.');
    if (playerStatus) playerStatus.textContent = 'Recompute done';
  } catch (e) {
    addLog('Recompute error: ' + e);
    if (playerStatus) playerStatus.textContent = 'Recompute error';
  }
}

root.addEventListener('click', function (e) {
  const btn = e.target.closest('button[data-action]');
  if (!btn) return;
  const action = btn.getAttribute('data-action');
  if (action === 'play') { _startUnifiedPlayback(1.0); return; }
  if (action === 'play-half') { playAtRate(0.5); return; }
  if (action === 'play-quarter') { playAtRate(0.25); return; }
  if (action === 'mark-start') { markVideo('start'); return; }
  if (action === 'mark-end') { markVideo('end'); return; }
  if (action === 'recompute') { recomputeMarked(); return; }
  if (action === 'prev-frame') { stepFrame(-1); return; }
  if (action === 'next-frame') { stepFrame(+1); return; }
  if (action === 'stop') {
    if (!videoPlayer) return;
    _stopUnifiedPlayback();
    try { videoPlayer.pause(); } catch (err) {}
    try { videoPlayer.currentTime = 0; } catch (err) {}
    const bounds = _boundsCache || _computeTimelineBounds();
    if (bounds && bounds.globalStart != null) _setCursorHost(bounds.globalStart, { noSeekVideo: true });
    return;
  }
  if (action === 'back') {
    if (!videoPlayer) return;
    _stopUnifiedPlayback();
    try { videoPlayer.pause(); } catch (err) {}
    const bounds = _boundsCache || _computeTimelineBounds();
    const thCur = (typeof _cursorHost === 'number' && isFinite(_cursorHost)) ? Number(_cursorHost) : hostTimeFromVideoTime(videoPlayer.currentTime);
    const thNew = (typeof thCur === 'number' && isFinite(thCur)) ? (thCur - 2.0) : null;
    if (thNew != null && bounds && bounds.globalStart != null && bounds.globalEnd != null) {
      _setCursorHost(_clamp(thNew, bounds.globalStart, bounds.globalEnd));
    }
    return;
  }
  if (action === 'refresh') { loadJumpList(); return; }
  if (action === 'save-annotation') { handleSaveAnnotation(); return; }
  if (action === 'delete-jump') { handleDeleteJump(); return; }
});

if (videoPlayer) videoPlayer.addEventListener('timeupdate', () => {
  // When user scrubs the native controls, keep the plot cursor synced by HOST time.
  // Avoid fighting the unified playback loops: only follow timeupdate when not actively playing via our loop.
  if (!videoPlayer) return;
  if (playbackRAF || playbackVfcHandle) return;
  const th = hostTimeFromVideoTime(videoPlayer.currentTime);
  if (typeof th === 'number' && isFinite(th)) _setCursorHost(th, { noSeekVideo: true });
});

async function handleSaveAnnotation() {
    if (!selectedJump || typeof selectedJump.jump_id !== 'number') {
      addLog('No jump selected to annotate.');
      return;
    }
    const jumpId = selectedJump.jump_id;
    const name = (jumpNameInput ? (jumpNameInput.value || '') : '').trim();
    const note = (jumpNote ? (jumpNote.value || '') : '').trim();
    try {
      if (annotationStatus) annotationStatus.textContent = 'Saving...';
      const resp = await fetch('/annotations/by_jump_id/' + jumpId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, note })
      });
      if (!resp.ok) {
        if (annotationStatus) annotationStatus.textContent = 'Save failed';
        addLog('Save annotation failed (' + resp.status + ')');
        return;
      }
      const data = await resp.json();
      selectedJump.name = name;
      selectedJump.note = note;
      if (annotationStatus) annotationStatus.textContent = 'Saved';
      addLog(data.detail || 'Annotation saved');
      const idx = _findJumpIndexByJumpId(jumpId);
      if (idx >= 0) {
        if (jumps[idx]) {
          jumps[idx].name = name;
          jumps[idx].note = note;
        }
        _updateJumpListItemByIndex(idx);
      }
    } catch (e) {
      if (annotationStatus) annotationStatus.textContent = 'Save error';
      addLog('Save annotation error: ' + e);
    }
}

async function handleDeleteJump() {
    // Get all selected jump IDs (from list selection)
    const jumpIdsToDelete = Array.from(selectedJumpIds).filter(id => typeof id === 'number');
    
    // If none selected, fall back to single selection (detail panel)
    if (jumpIdsToDelete.length === 0) {
      if (!selectedJump || typeof selectedJump.jump_id !== 'number') {
        addLog('No jump selected to delete.');
        return;
      }
      jumpIdsToDelete.push(selectedJump.jump_id);
    }
    
    // Build confirmation message
    let confirmMsg = `Delete ${jumpIdsToDelete.length} selected jump(s) from DB?\n\n`;
    if (jumpIdsToDelete.length === 1) {
      const jump = jumps.find(j => j && j.jump_id === jumpIdsToDelete[0]);
      const name = jump && jump.name ? jump.name : '';
      const label = name ? `${name} (jump_id=${jumpIdsToDelete[0]}, event_id=${jump.event_id})` : `jump_id=${jumpIdsToDelete[0]}`;
      confirmMsg += label + '\n\n';
    } else {
      confirmMsg += `Jump IDs: ${jumpIdsToDelete.join(', ')}\n\n`;
    }
    confirmMsg += 'This will also delete their IMU samples and frame data. This cannot be undone.';
    
    if (!confirm(confirmMsg)) {
      return;
    }
    
    try {
      if (deleteStatus) deleteStatus.textContent = `Deleting ${jumpIdsToDelete.length} jump(s)...`;
      const resp = await fetch('/db/jumps/bulk_delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jump_ids: jumpIdsToDelete })
      });
      if (!resp.ok) {
        if (deleteStatus) deleteStatus.textContent = 'Delete failed';
        let errorMsg = 'Delete failed (' + resp.status + ')';
        try {
          const err = await resp.json();
          if (err && err.detail) errorMsg += ': ' + err.detail;
        } catch (e) {}
        addLog(errorMsg);
        return;
      }
      const data = await resp.json();
      const deletedCount = data.deleted_count || jumpIdsToDelete.length;
      const imuCount = data.imu_samples_deleted || 0;
      const frameCount = data.frames_deleted || 0;
      addLog(`Deleted ${deletedCount} jump(s), ${imuCount} IMU sample(s), and ${frameCount} frame(s)`);
      if (deleteStatus) deleteStatus.textContent = `Deleted ${deletedCount} jump(s)`;
      await loadJumpList();
      clearSelectionAndPlots();
    } catch (e) {
      if (deleteStatus) deleteStatus.textContent = 'Delete error';
      addLog('Delete error: ' + e);
    }
}

// Initial
redrawPlots(null);
loadJumpList();
  }
};
