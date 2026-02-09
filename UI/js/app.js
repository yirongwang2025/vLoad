/**
 * vLoad SPA – client-side router (B.3).
 * Uses hash routing when URL is not an app path (e.g. /static/shell.html);
 * uses pathname when shell is served at /, /jumps, etc. (after B.5).
 */
(function () {
  'use strict';

  var APP_ROUTES = ['/', '/jumps', '/devices', '/skaters', '/coaches'];
  var ROUTE_TO_PAGE = { '/': 'connect', '/jumps': 'jumps', '/devices': 'devices', '/skaters': 'skaters', '/coaches': 'coaches' };
  /** B.5: view content from fragment API when shell is served at /, /jumps, etc. */
  var ROUTE_TO_FRAGMENT = { '/': '/api/fragments/connect', '/jumps': '/api/fragments/jumps', '/devices': '/api/fragments/devices', '/skaters': '/api/fragments/skaters', '/coaches': '/api/fragments/coaches' };

  function getPathFromHash() {
    var hash = (window.location.hash || '#').replace(/^#\/?/, '') || '';
    if (hash === '' || hash === '/') return '/';
    if (hash.indexOf('/') !== 0) hash = '/' + hash;
    return hash;
  }

  function isAppPath(pathname) {
    return pathname === '/' || pathname === '/jumps' || pathname === '/devices' || pathname === '/skaters' || pathname === '/coaches';
  }

  function getCurrentRoute() {
    var pathname = window.location.pathname;
    if (isAppPath(pathname)) return pathname;
    return getPathFromHash();
  }

  function navigate(path) {
    var pathname = window.location.pathname;
    if (isAppPath(pathname)) {
      if (pathname !== path) {
        history.pushState({ route: path }, '', path);
      }
      render(path);
      return;
    }
    var hash = path === '/' ? '#/' : '#' + path;
    if (window.location.hash !== hash) {
      window.location.hash = hash;
    } else {
      render(path);
    }
  }

  function escapeHtml(text) {
    if (text == null) return '';
    var s = String(text);
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function escapeAttr(text) {
    if (text == null) return '';
    return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function extractPageContent(html) {
    try {
      var parser = new DOMParser();
      var doc = parser.parseFromString(html, 'text/html');
      var parts = [];
      var h2 = doc.body && doc.body.querySelector('h2');
      if (h2) parts.push(h2.outerHTML);
      var msg = doc.getElementById('messageContainer');
      if (msg) parts.push(msg.outerHTML);
      var page = doc.querySelector('.page') || doc.querySelector('.layout');
      if (page) parts.push(page.outerHTML);
      if (parts.length) return parts.join('');
      var body = doc.body;
      if (body) return body.innerHTML;
    } catch (e) {
      console.error('extractPageContent', e);
    }
    return html;
  }

  /** Run fragment scripts in order; wait for each external script to load before running the next (so ConnectPage.init etc. run after connect.js loads). */
  function runScriptsFromFragment(html) {
    return new Promise(function (resolve, reject) {
      try {
        var parser = new DOMParser();
        var doc = parser.parseFromString(html, 'text/html');
        var scripts = doc.querySelectorAll('script');
        var i = 0;

        function runNext() {
          if (i >= scripts.length) {
            resolve();
            return;
          }
          var s = scripts[i++];
          var src = s.getAttribute('src');
          if (src) {
            var alreadyLoaded = false;
            var existing = document.querySelectorAll('script[src]');
            for (var k = 0; k < existing.length; k++) {
              if (existing[k].getAttribute('src') === src) { alreadyLoaded = true; break; }
            }
            if (alreadyLoaded) {
              var needReload = false;
              if (src.indexOf('connect.js') !== -1 && typeof window.ConnectPage === 'undefined') needReload = true;
              if (src.indexOf('jumps.js') !== -1 && typeof window.JumpsPage === 'undefined') needReload = true;
              if (src.indexOf('devices.js') !== -1 && typeof window.DevicesPage === 'undefined') needReload = true;
              if (src.indexOf('skaters.js') !== -1 && typeof window.SkatersPage === 'undefined') needReload = true;
              if (src.indexOf('coaches.js') !== -1 && typeof window.CoachesPage === 'undefined') needReload = true;
              if (needReload) {
                var all = document.querySelectorAll('script[src]');
                for (var j = 0; j < all.length; j++) {
                  if (all[j].getAttribute('src') === src) { all[j].parentNode.removeChild(all[j]); break; }
                }
                alreadyLoaded = false;
              }
            }
            if (alreadyLoaded) {
              runNext();
              return;
            }
            var ext = document.createElement('script');
            ext.onload = runNext;
            ext.onerror = runNext;
            ext.src = src;
            document.body.appendChild(ext);
          } else {
            var inline = document.createElement('script');
            inline.textContent = s.textContent;
            document.body.appendChild(inline);
            runNext();
          }
        }
        runNext();
      } catch (e) {
        console.error('runScriptsFromFragment', e);
        reject(e);
      }
    });
  }

  function hydrateConnectView(appEl) {
    fetch('/api/skaters', { headers: { 'Accept': 'application/json' } })
      .then(function (res) { return res.ok ? res.json() : Promise.reject(new Error(res.status)); })
      .then(function (data) {
        var sel = appEl.querySelector('#skaterSelect');
        if (!sel) return;
        var skaters = data.skaters || [];
        sel.innerHTML = '<option value="">-- Select Skater --</option>';
        skaters.forEach(function (s) {
          var opt = document.createElement('option');
          opt.value = s.id;
          opt.textContent = (s.name || '').trim() || '';
          sel.appendChild(opt);
        });
      })
      .catch(function (err) { console.error('hydrateConnectView', err); });
  }

  function hydrateJumpsView(appEl) {
    fetch('/db/jumps?limit=200', { headers: { 'Accept': 'application/json' } })
      .then(function (res) { return res.ok ? res.json() : Promise.reject(new Error(res.status)); })
      .then(function (data) {
        var ul = appEl.querySelector('#jumpList');
        if (!ul) return;
        var jumps = data.jumps || [];
        var parts = [];
        for (var i = 0; i < Math.min(jumps.length, 200); i++) {
          var j = jumps[i];
          var jid = j.jump_id != null ? j.jump_id : j.id;
          if (jid == null) continue;
          var eid = j.event_id;
          var name = (j.name || '').trim() || (eid != null ? 'Jump ' + eid : 'Jump');
          var tPeak = j.t_peak;
          var timeLabel = '';
          if (typeof tPeak === 'number' && isFinite(tPeak)) {
            var d = new Date(tPeak * 1000);
            timeLabel = d.getUTCHours().toString().padStart(2, '0') + ':' + d.getUTCMinutes().toString().padStart(2, '0') + ':' + d.getUTCSeconds().toString().padStart(2, '0');
          }
          var label = name + (timeLabel ? ' (' + timeLabel + ')' : '');
          parts.push(
            '<li data-jump-id="' + escapeAttr(jid) + '" data-event-id="' + escapeAttr(eid != null ? eid : '') + '" data-name="' + escapeAttr(name) + '" data-t-peak="' + escapeAttr(tPeak != null ? tPeak : '') + '"><label>' + escapeHtml(label) + '</label></li>'
          );
        }
        ul.innerHTML = parts.join('');
      })
      .catch(function (err) { console.error('hydrateJumpsView', err); });
  }

  function loadView(route, appEl) {
    var url = ROUTE_TO_FRAGMENT[route] || (route === '/' ? '/' : route);
    appEl.innerHTML = '<div class="page"><p>Loading…</p></div>';
    return fetch(url, { headers: { 'Accept': 'text/html' } })
      .then(function (res) {
        if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
        return res.text();
      })
      .then(function (html) {
        appEl.innerHTML = extractPageContent(html);
        return runScriptsFromFragment(html);
      })
      .then(function () {
        if (route === '/') hydrateConnectView(appEl);
        var inits = { '/': 'ConnectPage', '/jumps': 'JumpsPage', '/devices': 'DevicesPage', '/skaters': 'SkatersPage', '/coaches': 'CoachesPage' };
        var page = inits[route];
        if (page && typeof window[page] !== 'undefined' && typeof window[page].init === 'function') {
          window[page].init();
        }
        /* Jumps: do not hydrate – script’s loadJumpList() builds list and attaches click handlers; hydrating would overwrite them. */
      })
      .catch(function (err) {
        console.error('loadView', route, err);
        appEl.innerHTML = '<div class="page"><p>Failed to load view: ' + (err.message || route) + '.</p><p><a href="' + (route === '/' ? '/' : route) + '">Open full page</a></p></div>';
      });
  }

  function render(route) {
    var app = document.getElementById('app');
    if (!app) return;
    var pageId = ROUTE_TO_PAGE[route] || '';
    if (typeof vLoad !== 'undefined' && vLoad.renderNav) {
      vLoad.renderNav(pageId);
    }
    replaceNavLinksWithHash();
    loadView(route, app);
  }

  function replaceNavLinksWithHash() {
    if (isAppPath(window.location.pathname)) return;
    var container = document.getElementById('navContainer');
    if (!container) return;
    var links = container.querySelectorAll('.nav a[href="/"], .nav a[href="/jumps"], .nav a[href="/devices"], .nav a[href="/skaters"], .nav a[href="/coaches"]');
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var path = a.getAttribute('href') || '/';
      a.setAttribute('href', path === '/' ? '#/' : '#' + path);
    }
  }

  function bootstrap() {
    var app = document.getElementById('app');
    if (!app) return;
    var route = getCurrentRoute();
    if (APP_ROUTES.indexOf(route) === -1) route = '/';
    render(route);
  }

  function onNavClick(e) {
    var a = e.target && e.target.closest ? e.target.closest('a') : null;
    if (!a || !a.href) return;
    var pathname = window.location.pathname;
    var href = (a.getAttribute('href') || '').trim();
    if (href.indexOf('#') === 0) {
      var path = (href === '#' || href === '#/' ? '/' : href.slice(1));
      if (path.indexOf('/') !== 0) path = '/' + path;
      e.preventDefault();
      navigate(path);
      return;
    }
    if (href === '/' || href === '/jumps' || href === '/devices' || href === '/skaters' || href === '/coaches') {
      if (pathname.indexOf('/static/') === 0 || !isAppPath(pathname)) {
        e.preventDefault();
        navigate(href);
      }
    }
  }

  function onHashChange() {
    if (!document.getElementById('app')) return;
    var route = getPathFromHash();
    if (APP_ROUTES.indexOf(route) === -1) route = '/';
    render(route);
  }

  function onPopState(e) {
    if (!document.getElementById('app')) return;
    var route = (e.state && e.state.route) || window.location.pathname;
    if (APP_ROUTES.indexOf(route) === -1) route = '/';
    render(route);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      bootstrap();
      document.body.addEventListener('click', onNavClick);
      window.addEventListener('hashchange', onHashChange);
      window.addEventListener('popstate', onPopState);
    });
  } else {
    bootstrap();
    document.body.addEventListener('click', onNavClick);
    window.addEventListener('hashchange', onHashChange);
    window.addEventListener('popstate', onPopState);
  }
})();
