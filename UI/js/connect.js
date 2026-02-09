(function () {
  'use strict';
  // Connect page: WebSocket, IMU plots, connect/disconnect, config (extracted from index.html for 3.2)

    // WebSocket: add ping + reconnect + heartbeat to diagnose freezes/stalls.
    let ws = null;
    let _wsReconnectAttempt = 0;
    let _wsPingTimer = null;
    let _hbTimer = null;
    let _lastWsMsgMs = 0;
    let _lastImuMsgMs = 0;
    let _imuMsgCount = 0;
    let _imuSampleCount = 0;
    let _hbLastImuMsgCount = 0;
    let _hbLastImuSampleCount = 0;
    let _hbLastMs = null;

    const LOG_MAX_LINES = 600;
    const HEARTBEAT_EVERY_MS = 5000;
    const WS_PING_EVERY_MS = 2000;
    const WS_RECONNECT_BASE_MS = 500;
    const WS_RECONNECT_MAX_MS = 5000;

    /** Element refs, populated in init() from the Connect page container (supports SPA re-render). */
    let refs = {};

    // Use vLoad.addLog which safely no-ops if #logBox is missing; avoids throws when logBox is null
    const addLog = (typeof window.vLoad !== 'undefined' && typeof window.vLoad.addLog === 'function')
      ? window.vLoad.addLog
      : function (msg) { try { console.log('[vLoad]', msg); } catch (e) {} };

    const maxPts = 150;  // number of points kept in history
    // Acceleration series
    const accX = [], accY = [], accZ = [];
    // Gyro series
    const gyroX = [], gyroY = [], gyroZ = [];
    // Magnetometer series
    const magX = [], magY = [], magZ = [];
    // Host epoch timestamps for each plotted point (best-effort)
    const tHost = [];

    const colors = ['#1976d2', '#d32f2f', '#388e3c']; // x=blue, y=red, z=green
    let sampleRate = null; // Hz, from server messages
    let lastDrawTs = 0;    // throttle drawing to avoid overloading CPU
    let plotDirty = false; // only redraw when new samples arrive
    let _imuSampleCounter = 0; // for plot decimation
    const TARGET_PLOT_HZ = 80; // cap plotted sample rate to keep UI responsive (especially w/ video)
    let _lastPlotTHost = null;

    function wsStateLabel(wsObj) {
      if (!wsObj) return 'none';
      const rs = wsObj.readyState;
      if (rs === WebSocket.CONNECTING) return 'CONNECTING';
      if (rs === WebSocket.OPEN) return 'OPEN';
      if (rs === WebSocket.CLOSING) return 'CLOSING';
      if (rs === WebSocket.CLOSED) return 'CLOSED';
      return String(rs);
    }

    function stopWsTimers() {
      if (_wsPingTimer) { clearInterval(_wsPingTimer); _wsPingTimer = null; }
      if (_hbTimer) { clearInterval(_hbTimer); _hbTimer = null; }
    }

    function startWsTimers() {
      // Client -> server ping to keep server receive loop active.
      if (!_wsPingTimer) {
        _wsPingTimer = setInterval(() => {
          try {
            if (ws && ws.readyState === WebSocket.OPEN) ws.send('ping');
          } catch (e) {
            // ignore
          }
        }, WS_PING_EVERY_MS);
      }

      // Light heartbeat in the log.
      if (!_hbTimer) {
        _hbTimer = setInterval(() => {
          const now = Date.now();
          const dtMs = _hbLastMs == null ? null : (now - _hbLastMs);
          _hbLastMs = now;
          const wsState = wsStateLabel(ws);
          const sinceWs = _lastWsMsgMs ? (now - _lastWsMsgMs) : null;
          const sinceImu = _lastImuMsgMs ? (now - _lastImuMsgMs) : null;
          const dMsgs = _imuMsgCount - _hbLastImuMsgCount;
          const dSamp = _imuSampleCount - _hbLastImuSampleCount;
          _hbLastImuMsgCount = _imuMsgCount;
          _hbLastImuSampleCount = _imuSampleCount;
          const wsPart = sinceWs == null ? 'ws_rx=never' : `ws_rx=${(sinceWs/1000).toFixed(1)}s ago`;
          const imuPart = sinceImu == null ? 'imu_rx=never' : `imu_rx=${(sinceImu/1000).toFixed(1)}s ago`;
          const dtPart = dtMs == null ? 'dt=?' : `dt=${(dtMs/1000).toFixed(2)}s`;
          const sampHz = dtMs == null || dtMs <= 0 ? null : (dSamp * 1000.0 / dtMs);
          const msgHz = dtMs == null || dtMs <= 0 ? null : (dMsgs * 1000.0 / dtMs);
          const ratePart = sampHz == null ? '' : `, imu_rate≈${sampHz.toFixed(1)}Hz, imu_msg_rate≈${msgHz.toFixed(1)}/s`;
          addLog(`[HB] ws=${wsState}, ${wsPart}, ${imuPart}, ${dtPart}, imu_msgs+${dMsgs}, imu_samples+${dSamp}${ratePart}`);
        }, HEARTBEAT_EVERY_MS);
      }
    }

    function scheduleWsReconnect() {
      _wsReconnectAttempt += 1;
      const delay = Math.min(WS_RECONNECT_MAX_MS, WS_RECONNECT_BASE_MS * Math.pow(1.6, _wsReconnectAttempt));
      addLog(`WebSocket reconnect scheduled in ${(delay/1000).toFixed(1)}s...`);
      setTimeout(() => {
        connectWebSocket();
      }, delay);
    }

    function connectWebSocket() {
      try {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
      } catch (e) {
        // ignore
      }

      stopWsTimers();
      ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');

      ws.onopen = () => {
        _wsReconnectAttempt = 0;
        addLog('WebSocket connected');
        startWsTimers();
      };
      ws.onclose = () => {
        addLog('WebSocket disconnected');
        stopWsTimers();
        scheduleWsReconnect();
      };
      ws.onerror = () => { addLog('WebSocket error'); };

      ws.onmessage = (ev) => {
        _lastWsMsgMs = Date.now();
        try {
          const msg = JSON.parse(ev.data);

          // Log-type messages
          if (msg.type === 'log' && typeof msg.msg === 'string') {
            addLog(msg.msg);
            return;
          }

          // Jump events (if detection enabled)
          if (msg.type === 'jump') {
            _jumpCount += 1;
            if (refs.jumpCountEl) refs.jumpCountEl.textContent = String(_jumpCount);
            // Keep it short to avoid spamming; details are on /jumps page
            addLog(`Jump detected (event_id=${msg.event_id ?? ''}, rev≈${(typeof msg.revolutions_est === 'number') ? msg.revolutions_est.toFixed(1) : ''})`);
            return;
          }

          // IMU data messages
          if (typeof msg.rate === 'number') {
            sampleRate = msg.rate;
          }
          // Prefer full packet samples for smoother plotting; fall back to first_sample.
          const samples = Array.isArray(msg.samples) ? msg.samples : (msg.first_sample ? [msg.first_sample] : []);
          if (samples.length) {
            _imuMsgCount += 1;
            _imuSampleCount += samples.length;
            _lastImuMsgMs = Date.now();
            // IMU is actually connected when we receive data
            if (!imuConnected) {
              imuConnected = true;
              updateConnectionStatus();
              addLog('IMU sensor connected (receiving data)');
            }
            const decim = Math.max(1, Math.round(((sampleRate || 0) > 0 ? sampleRate : 200) / TARGET_PLOT_HZ));
            for (const s of samples) {
              _imuSampleCounter += 1;
              if ((_imuSampleCounter % decim) !== 0) continue;
              // Use server-provided per-sample t (epoch seconds) when available; fall back to packet time.
              const th = (s && typeof s.t === 'number' && isFinite(s.t)) ? Number(s.t) : (typeof msg.t === 'number' ? Number(msg.t) : null);
              pushLimited(tHost, (typeof th === 'number' && isFinite(th)) ? th : null);
              if (typeof th === 'number' && isFinite(th)) _lastPlotTHost = th;
              if (s && Array.isArray(s.acc)) {
                const a = s.acc;
                pushLimited(accX, a[0] ?? 0);
                pushLimited(accY, a[1] ?? 0);
                pushLimited(accZ, a[2] ?? 0);
              }
              if (s && Array.isArray(s.gyro)) {
                const g = s.gyro;
                pushLimited(gyroX, g[0] ?? 0);
                pushLimited(gyroY, g[1] ?? 0);
                pushLimited(gyroZ, g[2] ?? 0);
              }
              if (s && Array.isArray(s.mag)) {
                const m = s.mag;
                pushLimited(magX, m[0] ?? 0);
                pushLimited(magY, m[1] ?? 0);
                pushLimited(magZ, m[2] ?? 0);
              }
            }
            plotDirty = true;
            try {
              if (refs.timeStatusEl && typeof _lastPlotTHost === 'number') {
                refs.timeStatusEl.textContent = `Last IMU t_host: ${formatTimeFromEpoch(_lastPlotTHost)}`;
              }
            } catch (e) {}
          }
        } catch(e) {
          console.error(e);
        }
      };
    }

    let _lastVideoState = null;   // {running, has_frame, error}
    let _lastSessionId = null;
    let _jumpCount = 0;

    let currentSkaterId = null;
    let allSkaters = [];
    
    // Track connection status separately for IMU and Video
    let imuConnected = false;
    let videoConnected = false;

    // Load skaters on page load (use server-preloaded list if present, else fetch or DOM)
    async function loadSkaters() {
      const skaterSelect = refs.skaterSelect;
      if (!skaterSelect) return;
      try {
        const preloaded = typeof window.__PRELOADED_SKATERS__ !== 'undefined' && Array.isArray(window.__PRELOADED_SKATERS__);
        if (preloaded) {
          allSkaters = window.__PRELOADED_SKATERS__;
        } else if (skaterSelect.options.length > 1) {
          // Server-rendered options already in DOM; build allSkaters from them
          allSkaters = [];
          for (let i = 1; i < skaterSelect.options.length; i++) {
            const opt = skaterSelect.options[i];
            allSkaters.push({ id: parseInt(opt.value, 10), name: opt.textContent || '' });
          }
        } else {
          const response = await fetch('/api/skaters');
          if (!response.ok) {
            addLog('Failed to load skaters');
            return;
          }
          const data = await response.json();
          allSkaters = Array.isArray(data.skaters) ? data.skaters : [];
        }

        // Populate dropdown only if we don't already have server-rendered options
        if (skaterSelect.options.length <= 1) {
          skaterSelect.innerHTML = '<option value="">-- Select Skater --</option>';
          allSkaters.forEach(skater => {
            const option = document.createElement('option');
            option.value = skater.id;
            option.textContent = skater.name;
            skaterSelect.appendChild(option);
          });
        }

        // Restore last selected skater from localStorage
        try {
          const last = localStorage.getItem('vload_last_skater_id');
          if (last) {
            skaterSelect.value = last;
            currentSkaterId = parseInt(last);
            await loadSkaterSettings(currentSkaterId);
          }
        } catch (e) {
          console.error('localStorage error', e);
        }
      } catch (e) {
        console.error('Failed to load skaters:', e);
        addLog('Failed to load skaters: ' + e.message);
      }
    }

    // Load detection settings for selected skater
    async function loadSkaterSettings(skaterId) {
      if (!skaterId) {
        // Load defaults
        await loadDetectConfig();
        return;
      }
      try {
        const response = await fetch(`/api/skaters/${skaterId}/detection-settings`);
        if (!response.ok) {
          addLog('Failed to load skater settings');
          await loadDetectConfig(); // Fall back to defaults
          return;
        }
        const data = await response.json();
        const settings = data.settings || data.defaults || {};
        
        // Populate inputs
        const mh = refs.minHeightInput, ma = refs.minAzInput, mg = refs.minGzInput, ms = refs.minSepInput, mr = refs.minRevsInput;
        if (typeof settings.min_jump_height_m === 'number' && mh) mh.value = settings.min_jump_height_m.toFixed(2);
        if (typeof settings.min_jump_peak_az_no_g === 'number' && ma) ma.value = settings.min_jump_peak_az_no_g.toFixed(1);
        if (typeof settings.min_jump_peak_gz_deg_s === 'number' && mg) mg.value = settings.min_jump_peak_gz_deg_s.toFixed(0);
        if (typeof settings.min_new_event_separation_s === 'number' && ms) ms.value = settings.min_new_event_separation_s.toFixed(1);
        if (typeof settings.min_revs === 'number' && mr) mr.value = settings.min_revs.toFixed(1);
      } catch (e) {
        console.error('Failed to load skater settings:', e);
        await loadDetectConfig(); // Fall back to defaults
      }
    }

    // Handle skater selection change (bound in init)
    function bindSkaterSelect() {
      const skaterSelect = refs.skaterSelect;
      if (!skaterSelect) return;
      skaterSelect.onchange = async function() {
      const skaterId = parseInt(skaterSelect.value) || null;
      currentSkaterId = skaterId;
      if (skaterId) {
        try {
          localStorage.setItem('vload_last_skater_id', String(skaterId));
        } catch (e) {
          console.error('localStorage set error', e);
        }
        await loadSkaterSettings(skaterId);
      } else {
        await loadDetectConfig(); // Load defaults
      }
    };
    }

    function pushLimited(arr, value) {
      arr.push(value);
      while (arr.length > maxPts) arr.shift();
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
      // Show wall-clock time range if available (helps spot sync drift quickly).
      try {
        const t0 = tHost.length ? tHost[0] : null;
        const t1 = tHost.length ? tHost[tHost.length - 1] : null;
        if (typeof t0 === 'number' && isFinite(t0) && typeof t1 === 'number' && isFinite(t1) && t1 >= t0) {
          timeLabel += `  ${formatTimeFromEpoch(t0)} → ${formatTimeFromEpoch(t1)}`;
        }
      } catch (e) {}
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
      // Only redraw when data changed; keeps CPU low while video decoding is active.
      if (!plotDirty) {
        requestAnimationFrame(draw);
        return;
      }
      // Lower draw FPS when video is running (MJPEG decode can be heavy on the main thread).
      const videoRunning = _lastVideoState && _lastVideoState.running;
      const minIntervalMs = videoRunning ? 66 : 33; // ~15fps with video, ~30fps otherwise
      if (now - lastDrawTs < minIntervalMs) {
        requestAnimationFrame(draw);
        return;
      }
      lastDrawTs = now;
      plotDirty = false;
      if (refs.ctxAcc) drawSeries(refs.ctxAcc, [accX, accY, accZ]);
      if (refs.ctxGyro) drawSeries(refs.ctxGyro, [gyroX, gyroY, gyroZ]);
      if (refs.ctxMag) drawSeries(refs.ctxMag, [magX, magY, magZ]);
      requestAnimationFrame(draw);
    }

    // Helper function to update connection status message
    function updateConnectionStatus() {
      const connectionStatus = refs.connectionStatus;
      if (!connectionStatus) return;
      const parts = [];
      if (imuConnected) {
        parts.push('IMU: Connected');
      } else {
        parts.push('IMU: Not Available');
      }
      if (videoConnected) {
        parts.push('Video: Connected');
      } else {
        parts.push('Video: Not Available');
      }
      connectionStatus.textContent = parts.join(', ');
      if (imuConnected && videoConnected) {
        connectionStatus.style.color = '#388e3c'; // Green - both connected
      } else if (imuConnected || videoConnected) {
        connectionStatus.style.color = '#f57c00'; // Orange - partial connection
      } else {
        connectionStatus.style.color = '#d32f2f'; // Red - neither connected
      }
    }

    window.ConnectPage = {
      init: function () {
        const root = document.querySelector('[data-page="connect"]') || document.getElementById('app');
        if (!root) return;
        refs.root = root;
        refs.skaterSelect = root.querySelector('#skaterSelect');
        refs.connectBtn = root.querySelector('#connectBtn');
        refs.disconnectBtn = root.querySelector('#disconnectBtn');
        refs.connectionStatus = root.querySelector('#connectionStatus');
        refs.startDetectBtn = root.querySelector('#startDetectBtn');
        refs.stopDetectBtn = root.querySelector('#stopDetectBtn');
        refs.detectStatus = root.querySelector('#detectStatus');
        refs.jumpCountEl = root.querySelector('#jumpCount');
        refs.minHeightInput = root.querySelector('#minHeightInput');
        refs.minAzInput = root.querySelector('#minAzInput');
        refs.minGzInput = root.querySelector('#minGzInput');
        refs.minSepInput = root.querySelector('#minSepInput');
        refs.minRevsInput = root.querySelector('#minRevsInput');
        refs.saveConfigBtn = root.querySelector('#saveConfigBtn');
        refs.videoFeed = root.querySelector('#videoFeed');
        refs.logBox = root.querySelector('#logBox');
        refs.canvasAcc = root.querySelector('#plotAcc');
        refs.canvasGyro = root.querySelector('#plotGyro');
        refs.canvasMag = root.querySelector('#plotMag');
        refs.ctxAcc = refs.canvasAcc ? refs.canvasAcc.getContext('2d') : null;
        refs.ctxGyro = refs.canvasGyro ? refs.canvasGyro.getContext('2d') : null;
        refs.ctxMag = refs.canvasMag ? refs.canvasMag.getContext('2d') : null;
        refs.timeStatusEl = root.querySelector('#timeStatus');
        refs.videoTimeStatusEl = root.querySelector('#videoTimeStatus');

        if (refs.videoFeed) {
          refs.videoFeed.onerror = function () {
            addLog('Video preview error (failed to load /video/mjpeg). Try opening /video/snapshot.jpg and /video/debug.');
          };
        }

        // Event delegation for buttons (data-action)
        root.addEventListener('click', function (e) {
          const btn = e.target.closest('button[data-action]');
          if (!btn) return;
          const action = btn.getAttribute('data-action');
          if (action === 'connect') handleConnect();
          else if (action === 'disconnect') handleDisconnect();
          else if (action === 'save-config') handleSaveConfig();
          else if (action === 'start-detect') handleStartDetect();
          else if (action === 'stop-detect') handleStopDetect();
        });

        bindSkaterSelect();
        requestAnimationFrame(draw);
        connectWebSocket();

        // Initialize connection status
        updateConnectionStatus();
        setInterval(refreshVideoStatus, 1500);
        refreshVideoStatus();
        setInterval(function () {
          const now = Date.now();
          const IMU_TIMEOUT_MS = 3000;
          if (_lastImuMsgMs > 0 && (now - _lastImuMsgMs) < IMU_TIMEOUT_MS) {
            if (!imuConnected) { imuConnected = true; updateConnectionStatus(); }
          } else if (_lastImuMsgMs > 0 && (now - _lastImuMsgMs) >= IMU_TIMEOUT_MS) {
            if (imuConnected) { imuConnected = false; updateConnectionStatus(); addLog('IMU sensor disconnected (no data received)'); }
          }
        }, 1000);
        loadSkaters();
        loadDetectConfig();
      }
    };

    async function handleConnect() {
            if (!currentSkaterId) {
              if (refs.connectionStatus) {
                refs.connectionStatus.textContent = 'Please select a skater first';
                refs.connectionStatus.style.color = '#d32f2f';
              }
              addLog('Connect clicked but no skater selected');
              return;
            }
            if (refs.connectionStatus) {
              refs.connectionStatus.textContent = 'Connecting...';
              refs.connectionStatus.style.color = '#666';
            }
            imuConnected = false;
            videoConnected = false;
            addLog(`Connect request for skater ID ${currentSkaterId}`);
      
            try {
              // Connect to IMU sensor
              const connectResp = await fetch('/connect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ skater_id: currentSkaterId })
              });
              
              if (!connectResp.ok) {
                const errorData = await connectResp.json().catch(() => ({ detail: 'Connection failed' }));
                const errorMsg = errorData.detail || `Connect failed with status ${connectResp.status}`;
                addLog(`IMU connect failed: ${errorMsg}`);
                imuConnected = false;
              } else {
                const connectData = await connectResp.json();
                addLog(connectData.detail || 'IMU connect request sent');
                // Don't set imuConnected = true here - wait for actual IMU data to arrive
                // The status will be updated when IMU messages are received via WebSocket
              }
      
              // Connect to camera (try regardless of IMU status)
              try {
                const videoResp = await fetch('/video/connect', { method: 'POST' });
                if (!videoResp.ok) {
                  addLog('Video connect failed with status ' + videoResp.status);
                  videoConnected = false;
                } else {
                  await videoResp.json();
                  addLog('Video connect request sent');
                  if (refs.videoFeed) {
                    refs.videoFeed.src = '/video/mjpeg?fps=10&ts=' + Date.now();
                  }
                  // Check actual video status - refreshVideoStatus will set videoConnected based on real status
                  await refreshVideoStatus();
                  // Don't set videoConnected = true here - refreshVideoStatus will set it based on actual status
                }
              } catch (e) {
                addLog('Video connect error: ' + e);
                videoConnected = false;
              }
              
              // Update status message with both IMU and Video status
              updateConnectionStatus();
            } catch (e) {
              console.error(e);
              imuConnected = false;
              videoConnected = false;
              if (refs.connectionStatus) {
                refs.connectionStatus.textContent = 'Connection error: ' + e.message;
                refs.connectionStatus.style.color = '#d32f2f';
              }
              addLog(`Connect error: ${e}`);
            }
    }

    async function handleDisconnect() {
            if (refs.connectionStatus) {
              refs.connectionStatus.textContent = 'Disconnecting...';
              refs.connectionStatus.style.color = '#666';
            }
            addLog('Disconnect button clicked');
            try {
              // Disconnect IMU
              const resp = await fetch('/disconnect', { method: 'POST' });
              if (!resp.ok) {
                addLog(`IMU disconnect failed with status ${resp.status}`);
              } else {
                const data = await resp.json();
                addLog(data.detail || 'IMU disconnect request sent');
              }
              imuConnected = false;
      
              // Disconnect video
              try {
                const videoResp = await fetch('/video/disconnect', { method: 'POST' });
                if (!videoResp.ok) {
                  addLog('Video disconnect failed with status ' + videoResp.status);
                } else {
                  await videoResp.json();
                  addLog('Video streaming stopped');
                  if (refs.videoFeed) {
                    refs.videoFeed.removeAttribute('src');
                  }
                  await refreshVideoStatus();
                }
              } catch (e) {
                addLog('Video disconnect error: ' + e);
              }
              videoConnected = false;
      
              // Update status message
              updateConnectionStatus();
            } catch (e) {
              console.error(e);
              addLog(`Disconnect error: ${e}`);
              imuConnected = false;
              videoConnected = false;
              updateConnectionStatus();
            }
    }

    async function refreshVideoStatus() {
            try {
              const resp = await fetch('/video/status');
              if (!resp.ok) {
                videoConnected = false;
                updateConnectionStatus();
                return;
              }
              const st = await resp.json();
              const cur = { running: !!st.running, has_frame: !!st.has_frame, error: st.error || null };
              const changed = !_lastVideoState ||
                cur.running !== _lastVideoState.running ||
                cur.has_frame !== _lastVideoState.has_frame ||
                cur.error !== _lastVideoState.error;
              if (changed) {
                let text = `Video status: ${cur.running ? 'connected' : 'disconnected'}`;
                if (cur.running && cur.has_frame) text += ', frames ok';
                if (cur.error) text += `, error: ${cur.error}`;
                addLog(text);
                _lastVideoState = cur;
              }
              
              // Update video connection status
              videoConnected = cur.running;
              updateConnectionStatus();
      
              // Host-time only: show latest frame timestamp if available.
              try {
                if (refs.videoTimeStatusEl) {
                  const th = (typeof st.t_last_frame === 'number' && isFinite(st.t_last_frame)) ? Number(st.t_last_frame) : null;
                  refs.videoTimeStatusEl.textContent = th != null ? `Last video frame t_host: ${formatTimeFromEpoch(th)}` : '';
                }
              } catch (e) {}
            } catch (e) {
              videoConnected = false;
              updateConnectionStatus();
              // ignore
            }
          }

    async function loadDetectConfig() {
            try {
              const resp = await fetch('/config');
              if (!resp.ok) {
                addLog('Config load failed with status ' + resp.status);
                return;
              }
              const data = await resp.json();
              const jc = data.jump || {};
              const mh = refs.minHeightInput, ma = refs.minAzInput, mg = refs.minGzInput, ms = refs.minSepInput, mr = refs.minRevsInput;
              if (typeof jc.min_jump_height_m === 'number' && mh) mh.value = jc.min_jump_height_m.toFixed(2);
              if (typeof jc.min_jump_peak_az_no_g === 'number' && ma) ma.value = jc.min_jump_peak_az_no_g.toFixed(1);
              if (typeof jc.min_jump_peak_gz_deg_s === 'number' && mg) mg.value = jc.min_jump_peak_gz_deg_s.toFixed(0);
              if (typeof jc.min_new_event_separation_s === 'number' && ms) ms.value = jc.min_new_event_separation_s.toFixed(1);
              if (typeof jc.min_revs === 'number' && mr) mr.value = jc.min_revs.toFixed(1);
            } catch (e) {
              console.error(e);
              addLog('Config load error: ' + e);
            }
          }

    async function handleSaveConfig() {
              if (!currentSkaterId) {
                addLog('Please select a skater first');
                return;
              }
              const mh = refs.minHeightInput, ma = refs.minAzInput, mg = refs.minGzInput, ms = refs.minSepInput, mr = refs.minRevsInput;
              const payload = {
                min_jump_height_m: mh ? parseFloat(mh.value) : 0,
                min_jump_peak_az_no_g: ma ? parseFloat(ma.value) : 0,
                min_jump_peak_gz_deg_s: mg ? parseFloat(mg.value) : 0,
                min_new_event_separation_s: ms ? parseFloat(ms.value) : 0,
                min_revs: mr ? parseFloat(mr.value) : 0,
              };
              addLog(`Saving detection settings for skater ${currentSkaterId}...`);
              try {
                const resp = await fetch(`/api/skaters/${currentSkaterId}/detection-settings`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ jump: payload })
                });
                if (!resp.ok) {
                  const errorData = await resp.json().catch(() => ({ detail: 'Save failed' }));
                  addLog('Config save failed: ' + (errorData.detail || resp.status));
                  return;
                }
                const data = await resp.json();
                addLog(data.detail || 'Detection settings saved for skater');
              } catch (e) {
                console.error(e);
                addLog('Config save error: ' + e);
              }
    }

    async function handleStartDetect() {
              if (refs.detectStatus) refs.detectStatus.textContent = 'Enabling...';
              addLog('Detection start requested...');
              try {
                const resp = await fetch('/detection/start', { method: 'POST' });
                if (!resp.ok) {
                  if (refs.detectStatus) refs.detectStatus.textContent = 'Error: ' + resp.status;
                  addLog('Detection start failed with status ' + resp.status);
                  return;
                }
                const data = await resp.json();
                if (refs.detectStatus) refs.detectStatus.textContent = data.detail || 'Detection enabled';
                addLog(data.detail || 'Jump detection enabled');
              } catch (e) {
                console.error(e);
                if (refs.detectStatus) refs.detectStatus.textContent = 'Error';
                addLog('Detection start error: ' + e);
              }
    }

    async function handleStopDetect() {
              if (refs.detectStatus) refs.detectStatus.textContent = 'Disabling...';
              addLog('Detection stop requested...');
              try {
                const resp = await fetch('/detection/stop', { method: 'POST' });
                if (!resp.ok) {
                  if (refs.detectStatus) refs.detectStatus.textContent = 'Error: ' + resp.status;
                  addLog('Detection stop failed with status ' + resp.status);
                  return;
                }
                const data = await resp.json();
                if (refs.detectStatus) refs.detectStatus.textContent = data.detail || 'Detection disabled';
                addLog(data.detail || 'Jump detection disabled');
              } catch (e) {
                console.error(e);
                if (refs.detectStatus) refs.detectStatus.textContent = 'Error';
                addLog('Detection stop error: ' + e);
              }
    }
})();
