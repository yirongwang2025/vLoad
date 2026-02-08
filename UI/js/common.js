/**
 * vLoad shared UI – API helpers, nav, optional logger and WebSocket.
 * Include before page scripts: <script src="/static/js/common.js"></script>
 */

(function (global) {
  'use strict';

  const NAV_LINKS = [
    { id: 'connect', href: '/', label: 'Connect' },
    { id: 'jumps', href: '/jumps', label: 'Jump Review' },
    { id: 'devices', href: '/devices', label: 'Device Management' },
    { id: 'skaters', href: '/skaters', label: 'Skater Management' },
    { id: 'coaches', href: '/coaches', label: 'Coach Management' },
  ];

  /**
   * Renders the shared top nav into the given container.
   * @param {string} currentPage - One of: 'connect' | 'jumps' | 'devices' | 'skaters' | 'coaches'
   * @param {string} [containerId='navContainer'] - ID of the element to render into.
   */
  function renderNav(currentPage, containerId) {
    const id = containerId || 'navContainer';
    const el = document.getElementById(id);
    if (!el) return;
    const active = (currentPage || '').toLowerCase();
    let html = '<div class="nav">';
    NAV_LINKS.forEach(function (link) {
      const cls = link.id === active ? 'active' : '';
      html += '<a href="' + escapeHtml(link.href) + '"' + (cls ? ' class="' + cls + '"' : '') + '>' + escapeHtml(link.label) + '</a>';
    });
    html += '</div>';
    el.innerHTML = html;
  }

  function escapeHtml(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * GET request; returns parsed JSON or throws.
   * @param {string} path - Path (e.g. '/api/skaters').
   * @returns {Promise<object>}
   */
  async function apiGet(path) {
    const res = await fetch(path);
    if (!res.ok) {
      const err = await res.json().catch(function () { return { detail: res.statusText }; });
      throw new Error(err.detail || 'Request failed ' + res.status);
    }
    return res.json();
  }

  /**
   * POST request with JSON body; returns parsed JSON or throws.
   * @param {string} path
   * @param {object} body
   * @returns {Promise<object>}
   */
  async function apiPost(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      const err = await res.json().catch(function () { return { detail: res.statusText }; });
      throw new Error(err.detail || 'Request failed ' + res.status);
    }
    return res.json();
  }

  const LOG_MAX_LINES = 600;

  /**
   * Appends a line to the page log box if present (#logBox).
   * @param {string} msg
   */
  function addLog(msg) {
    const box = document.getElementById('logBox');
    if (!box) return;
    const ts = new Date().toISOString();
    box.textContent += '[' + ts + '] ' + msg + '\n';
    const lines = box.textContent.split('\n');
    if (lines.length > LOG_MAX_LINES) {
      box.textContent = lines.slice(lines.length - LOG_MAX_LINES).join('\n');
    }
    box.scrollTop = box.scrollHeight;
  }

  /**
   * Thin WebSocket helper: create WS, optional onMessage(parsed), onOpen, onClose, onError.
   * @param {string} path - Path (e.g. '/ws').
   * @param {{ onMessage?: (data: object) => void, onOpen?: () => void, onClose?: () => void, onError?: () => void }} [handlers]
   * @returns {WebSocket}
   */
  function createWs(path, handlers) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(protocol + '//' + location.host + (path || '/ws'));
    if (handlers && handlers.onOpen) ws.onopen = handlers.onOpen;
    if (handlers && handlers.onClose) ws.onclose = handlers.onClose;
    if (handlers && handlers.onError) ws.onerror = handlers.onError;
    if (handlers && handlers.onMessage) {
      ws.onmessage = function (ev) {
        try {
          const data = JSON.parse(ev.data);
          handlers.onMessage(data);
        } catch (e) {
          // ignore
        }
      };
    }
    return ws;
  }

  global.vLoad = global.vLoad || {};
  global.vLoad.renderNav = renderNav;
  global.vLoad.apiGet = apiGet;
  global.vLoad.apiPost = apiPost;
  global.vLoad.addLog = addLog;
  global.vLoad.createWs = createWs;
  global.vLoad.escapeHtml = escapeHtml;

  // Auto-render nav when container exists and is empty (so nav shows even if page script never runs)
  if (typeof document !== 'undefined') {
    var container = document.getElementById('navContainer');
    if (container && !container.innerHTML) {
      var page = (document.body && document.body.getAttribute('data-nav-page')) || 'connect';
      renderNav(page);
    }
  }
})(typeof window !== 'undefined' ? window : this);
