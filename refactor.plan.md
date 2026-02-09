# Refactor Plan

This document outlines a phased refactoring plan for the vLoad codebase (modules, UI, and server.py), based on the suggested refactor order. Work is ordered by risk and impact: low-risk clarity wins first, then structural changes, then larger architectural shifts.

---

## Phase 1: Low risk, high clarity

### 1.1 Shared UI assets

**Goal:** Reduce duplication and drift across HTML pages.

- [x] **Shared CSS**
  - Add `UI/css/common.css` with: nav, layout, panels, buttons, messages, form groups.
  - Each page: `<link rel="stylesheet" href="/static/css/common.css">` (or equivalent if templates inject it).
  - Remove or trim duplicated `<style>` blocks from index.html, devices.html, skaters.html, coaches.html, jumps.html so they only override page-specific rules.

- [x] **Shared nav**
  - Define nav markup in one place (e.g. a small partial template, or a shared JS snippet that accepts "current page" and renders the same nav).
  - Ensure all five pages use it so adding a new top-level link is a single change.

- [x] **Shared JS (optional but recommended)**
  - Add `UI/js/common.js` with: `apiGet(path)`, `apiPost(path, body)`, optional `addLog(msg)` or logger, and optionally a thin WebSocket helper.
  - Pages include `<script src=".../common.js">` and use these helpers instead of reimplementing fetch/error handling.

**Deliverables:** `UI/css/common.css`, shared nav usage in all pages, optionally `UI/js/common.js`. No URL or API changes.

---

### 1.2 Split server.py into routers

**Goal:** Group routes by domain without changing URLs or behavior.

- [x] **Routers**
  - `routers/pages.py` – HTML page handlers (and `load_html_template` if it stays server-side).
  - `routers/api_devices.py` – All `/api/devices` and `/scan` routes.
  - `routers/api_skaters.py` – All `/api/skaters` routes (including detection-settings).
  - `routers/api_coaches.py` – All `/api/coaches` and skater–coach link routes.
  - `routers/api_jumps.py` – All `/db/jumps*` routes (list, get, delete, marks, recompute, pose, bulk_delete).
  - `routers/video.py` – `/video/connect`, `/video/disconnect`, `/video/status`, `/video/mjpeg`, `/video/snapshot.jpg`, `/video/debug`.
  - `routers/sessions.py` – `/session/start`, `/session/stop`, `/session/status`; `/sessions/{id}/video`, `/sessions/{id}/frames`, `/debug/status`.
  - `routers/ws.py` – WebSocket endpoint and `ConnectionManager` (manager in `routers/ws.py`, referenced via `state.manager`).

- [x] **Mounting**
  - In `server.py`, FastAPI app created; all routers included with `app.include_router(...)`. No path prefixes; existing URLs unchanged.

- [x] **Shared dependencies**
  - Globals used by routes live in `server.py` or `state.py`; routers import `state` for shared refs (`_dbg`, `_session_id`, `_session_dir`, templates, helpers). No state migration in this phase.

**Deliverables:** New `routers/` package, `server.py` slimmed to app creation, lifespan, config, and router includes. All existing routes still work (duplicate handlers removed; `session_start`/`session_stop` imported from `routers.sessions` for `/detection/*`).

---

### 1.3 Pydantic request/response models

**Goal:** Document and validate a few high-traffic endpoints.

- [x] **Models**
  - Added Pydantic models in `schemas/requests.py`: `ConnectPayload` (device, skater_id, mode, rate), `SessionStartPayload` (session_id), `JumpMarksPayload` (which/kind, t_host, t_video), `BulkDeletePayload` (jump_ids, min_length=1). Optional `schemas/responses.py` for connect/session response docs.

- [x] **Use in routes**
  - Replaced `payload: Dict[str, Any]` with the appropriate model: `/connect` → `ConnectPayload`, `/session/start` → `SessionStartPayload`, `/db/jumps/{event_id}/marks` and `/db/jumps/by_jump_id/{jump_id}/marks` → `JumpMarksPayload`, `/db/jumps/bulk_delete` → `BulkDeletePayload`. All fields optional with defaults except `jump_ids` (required list).

**Deliverables:** New `schemas/` package (requests.py, responses.py), updated route signatures. Validation and OpenAPI docs improve; behavior unchanged.

---

## Phase 2: Medium effort

### 2.1 Split db.py by domain

**Goal:** Break the 2,700+ line module into maintainable pieces.

- [x] **New modules**
  - `modules/db/pool.py` – Pool creation, `init_db()`, `get_status()`, `close_db()`, `get_pool()`, `_to_dt()`, and all CREATE TABLE logic.
  - `modules/db/jumps.py` – Jumps, jump_frames, imu_samples: insert_jump_with_imu, list_jumps, get_jump_with_imu*, update_jump_video_mark*, recompute_marked_imu_metrics*, delete_jump*, delete_jumps_bulk, set_jump_video_path*, replace_jump_frames, get_jump_frames, update_jump_pose_metrics, resolve_jump_row_id, annotations.
  - `modules/db/sessions.py` – Sessions, frames: upsert_session_start, update_session_stop, replace_frames, get_frames, update_session_camera_calibration.
  - `modules/db/devices.py` – Devices: list_devices, get_device_by_mac, get_device_by_name, upsert_device, delete_device, resolve_device_identifier.
  - `modules/db/skaters.py` – Skaters, skater_devices, skater_coaches, detection settings: list_skaters, get_skater_by_id, upsert_skater, delete_skater, get_skater_devices, add/remove_skater_device, get_skater_coaches, add/remove_skater_coach, get_skater_detection_settings, upsert_skater_detection_settings.
  - `modules/db/coaches.py` – Coaches: list_coaches, get_coach_by_id, upsert_coach, delete_coach, get_coach_skaters.
  - `modules/db/__init__.py` – Re-exports of all public functions so `from modules import db` and `db.list_jumps(...)` etc. still work.

- [x] **Shared helpers**
  - Add helpers for row→dict mapping (e.g. `jump_row_to_dict`, `device_row_to_dict`, timestamp/float extraction) in `modules/db/helpers.py` to avoid repeating the same 30+ field logic in every list/get.

- [x] **Migration**
  - Monolith `modules/db.py` renamed to `modules/db_legacy.py`. The package `modules/db/` is the single entry point; `from modules import db` loads the package. No change to public `db.*` API used by server/routers.

**Deliverables:** `modules/db/` package with domain modules and a single entry point. No change to public `db.*` API used by server/routers.

---

### 2.2 Unify jump API (by jump_id)

**Goal:** One set of jump operations keyed by `jump_id`; support `event_id` only where needed via resolution.

- [x] **API design**
  - Treat `jump_id` (jumps.id) as the canonical identifier for all jump operations.
  - Canonical routes: GET/DELETE `/db/jumps/{jump_id}`, POST `/db/jumps/{jump_id}/marks`, `/recompute_marked_metrics`, `/pose_metrics`, POST `/pose/jumps/{jump_id}/run`. Deprecated event_id wrappers at `/db/jumps/by_event_id/{event_id}` (and same for marks, recompute, pose_metrics, pose run) resolve and delegate.

- [x] **DB layer**
  - In `modules/db/jumps.py`, added `resolve_jump_id_from_event_id(event_id)` and `update_jump_pose_metrics_by_jump_id(jump_id, ...)`. Existing by_jump_id functions are the canonical implementations; callers that need “by event_id”, add a thin `resolve_jump_id(event_id)` (or reuse existing resolution) and a small wrapper that resolves then calls the jump_id path.

- [x] **Server/routers**
  - `routers/api_jumps.py`: canonical jump_id routes; by_event_id routes registered before `{jump_id}` so path matching is correct; they resolve and delegate to canonical handlers.

- [x] **UI**
  - jumps.html uses `/db/jumps/{jump_id}` for get, marks, recompute; `/annotations/by_jump_id/{jump_id}` and `/db/jumps/bulk_delete` unchanged.

**Deliverables:** Single set of jump endpoints by jump_id; event_id compatibility at `/db/jumps/by_event_id/*`; DB layer with resolve_jump_id_from_event_id and update_jump_pose_metrics_by_jump_id.

---

### 2.3 Lazy or static HTML loading

**Goal:** Server can run without UI files; avoid load-time failures for missing templates.

*Note: Server routing (Phase 1.2) is already in place and independent of this step. Either option below works with the current routers.*

- [ ] **Option A – Lazy load**
  - In `load_html_template`, read the file when first requested (e.g. cache per filename in a dict or on the route). Remove module-level `INDEX_HTML = load_html_template(...)` so server starts even if `UI/` is missing; 404 or 500 only when a page is actually requested.

- [ ] **Option B – Static + SPA** (recommended for multi-device: computers, pads, phones)
  - One codebase and same URLs for all devices; responsive layout; no full reloads. Refactor steps:
  - [x] **B.1 – Remove HTML load at server import**
    - In `server.py`: remove module-level `load_html_template(...)` for index.html, jumps.html, devices.html, skaters.html, coaches.html and the assignments into `state.INDEX_HTML`, `state.JUMPS_HTML`, etc. Keep `load_html_template` (or a lazy reader) only for serving the single SPA shell when needed. *Done: added `get_page_html()` with cache; pages use it via `state.get_page_html`; 404 when template missing.*
  - [x] **B.2 – Single SPA entry and static serving**
    - One shell: `UI/index.html` – loads `common.css` and one main JS bundle (e.g. `UI/js/app.js`), root element (e.g. `<div id="app">`) for the client router. Continue serving `UI/` at `/static/` via existing `StaticFiles` mount in `server.py`. *Done: created `UI/shell.html` (minimal shell with common.css, nav, `#app`, common.js, app.js) and `UI/js/app.js` (bootstrap; routing in B.3). Current `index.html` remains the Connect page so `/` still works; in B.5 serve shell for app routes (or point to shell.html).*
  - [x] **B.3 – Client-side routing**
    - In `UI/js/app.js` (or main bundle): implement a small router (hash `#/jumps` or History API `/jumps`). Map routes to views: `/` → Connect, `/jumps` → Jump Review, `/devices`, `/skaters`, `/coaches`. Each view = current page logic inlined, or fragment/template fetch, or JS-rendered. *Done: router in app.js with hash routing (when on /static/shell.html) and pathname support (for B.5). Nav links rewritten to hash in shell; click/hashchange/popstate handled. Views loaded by fetching server-rendered page and injecting `.page` into #app (injected scripts do not run; add connect.js/jumps.js later for full behavior).*
  - [x] **B.4 – Replace server-injected preload with API**
    - Connect: SPA fetches skaters via existing API (e.g. `GET /api/skaters`) on mount instead of `SKATER_OPTIONS` / `__PRELOADED_SKATERS__`. Jumps: SPA fetches jumps list via existing API and renders list in client instead of `__PRELOADED_JUMPSS__` and server-rendered list markup. *Done: in app.js after injecting Connect view call `hydrateConnectView(appEl)` (GET /api/skaters, populate #skaterSelect); after injecting Jumps view call `hydrateJumpsView(appEl)` (GET /db/jumps?limit=200, render <li> items into #jumpList). Server preload left in place for full-page GET / and GET /jumps.*
  - [x] **B.5 – Serve SPA shell for app routes**
    - In `routers/pages.py`: add a catch-all (registered after API/static) so GET `/`, `/jumps`, `/devices`, `/skaters`, `/coaches` return the same SPA shell HTML (content of `UI/index.html`). Do not serve shell for `/static/*`, `/api/*`, `/video/*`, etc. Alternative: serve shell only at `/` and use hash routing (`/#/jumps`) so no catch-all needed. *Done: GET /, /jumps, /devices, /skaters, /coaches return shell (shell.html). Added GET /api/fragments/connect, /api/fragments/jumps, /api/fragments/devices, /api/fragments/skaters, /api/fragments/coaches for view HTML. app.js loadView fetches fragment URLs (ROUTE_TO_FRAGMENT) so view content loads correctly when shell is served. /static/index.html and /static/jumps.html still serve full pages (legacy).*
  - [x] **B.6 – Cleanup**
    - Remove or repurpose old per-page handlers that return different HTML; remove `/static/index.html` and `/static/jumps.html` overrides. Stop populating `state.INDEX_HTML`, `state.JUMPS_HTML`, etc. *Done: removed GET /static/index.html and GET /static/jumps.html handlers; those URLs are now served by StaticFiles (raw UI files). state.INDEX_HTML etc. were already removed in B.1 (only get_page_html in state).*
  - [x] **B.7 – Multi-device**
    - Use `UI/css/common.css` (and any kept CSS) for responsive breakpoints, touch-friendly targets, flexible layout. Same SPA and server routing serve all device types. *Done: common.css updated with box-sizing; responsive body padding; nav/buttons min-height 44px and touch-action; layout/row/col breakpoints at 900px and 600px (stack columns, full-width panels); form inputs min-height 44px and font-size 16px to avoid iOS zoom; jump list rows touch-friendly; responsive images/video; .topLeft/.topRight relax on small screens.*

**Deliverables (Option A):** No HTML read at server import time; clear error when a requested page file is missing.

**Deliverables (Option B):** Single SPA shell; client-side routing; app routes serve shell; preload replaced by API; server starts without UI; one codebase for multi-device.

---

## Phase 3: Larger architectural changes

### 3.1 Replace globals with explicit app state

**Goal:** Make dependencies and lifecycle explicit; simplify testing and reasoning.

- [x] **State object**
  - Introduce an `AppState` (or similar) dataclass or small module holding: IMU process handle, jump queue, jump_events, next_event_id, annotations, jump_windows_by_session, detection_enabled, debug counters, session_id, frame_history, imu_history, clip worker proc, video proc, etc.
  - Create one instance in lifespan and attach to `app.state` (e.g. `app.state.state = AppState()`).
  - *Done: `app_state.py` defines `AppState`; one instance created in lifespan and attached as `app.state.state`; also stored as `_app_state` for helpers outside request context.*

- [x] **Inject state**
  - Pass `request.app.state.state` (or a dependency that returns it) into route handlers and background tasks that need it. Refactor workers (e.g. jump_worker_loop, frame_sync_loop) to accept state as an argument instead of reading globals.
  - *Done: `deps.get_state(request)` used via `Depends(get_state)` in all relevant routes (server.py and routers: sessions, pages, api_jumps, video). Workers and UDP protocol take `AppState`; state module bridge removed.*

- [x] **Remove globals**
  - Once all reads/writes go through app.state, remove the global variables from server.py (and any other modules that currently use them).
  - *Done: All mutable runtime state removed from server.py. Config and constants (CFG, MODE, RATE, IMU_UDP_*, JUMP_CONFIG_DEFAULTS, etc.) kept at module level; video backend and locks created in lifespan on AppState.*

**Deliverables:** Single source of truth for runtime state; no global mutable state for app lifecycle; easier to test with a fresh state instance.

---

### 3.2 Extract page-specific JS into separate files

**Goal:** Shrink inline script blocks; enable reuse and caching.

- [x] **Connect page**
  - Move the bulk of index.html’s `<script>` into `UI/js/connect.js` (or `index.js`). Leave a minimal inline bootstrap that initializes the app (e.g. calls `ConnectPage.init()` or similar). Ensure WebSocket, plot, connect/disconnect, and config logic live in the JS file.
  - **Done:** `connect.js` exposes `ConnectPage.init()`; index.html loads common.js, connect.js, then calls `vLoad.renderNav('connect')` and `ConnectPage.init()`.

- [x] **Jump review page**
  - Move jumps.html’s script into `UI/js/jumps.js` with a small inline bootstrap. Keep list, detail, video/IMU sync, and delete logic in the file.
  - **Done:** `jumps.js` exposes `JumpsPage.init()`; jumps.html uses same bootstrap pattern.

- [x] **Other pages**
  - If devices/skaters/coaches have non-trivial JS, extract to `devices.js`, `skaters.js`, `coaches.js` as needed.
  - **Done:** `devices.js` (DevicesPage.init), `skaters.js` (SkatersPage.init), `coaches.js` (CoachesPage.init); each HTML has minimal bootstrap.

- [x] **Serving**
  - Serve JS (and CSS) from `/static/` or equivalent so browsers can cache them. Ensure paths in HTML match (e.g. `<script src="/static/js/connect.js">`).
  - **Done:** All pages use `/static/js/*.js` and `/static/css/common.css`.

**Deliverables:** Smaller HTML files; larger but cacheable JS (and shared CSS from Phase 1). No change to behavior.

---

### 3.3 Optional: Minimal front-end build step

**Goal:** Shared chunks and smaller per-page bundles if the project grows.

- [ ] **Tooling**
  - Add a minimal build (e.g. esbuild or a single concatenation script) that: (1) builds `common.js` from shared sources, (2) builds per-page bundles that may depend on common, (3) outputs to e.g. `UI/dist/` or `static/`.

- [ ] **Integration**
  - In development, either serve source files or run the build in watch mode. For production, serve the built assets. Keep the rest of the refactor build-agnostic so this step remains optional.

**Deliverables:** Optional build script and documented dev/prod flow. Can be skipped if the project stays small.

---

## Summary table

| Phase | Item | Risk | Effort | Outcome |
|-------|------|------|--------|---------|
| 1.1 | Shared UI CSS + nav + optional common.js | Low | Small | Less duplication; single nav/CSS source |
| 1.2 | Routers in server.py | Low | Medium | Clear route grouping; same URLs |
| 1.3 | Pydantic models for key payloads | Low | Small | Validation + docs |
| 2.1 | Split db.py by domain | Medium | Large | Maintainable DB layer |
| 2.2 | Unify jump API by jump_id | Medium | Medium | Single jump API surface |
| 2.3 | Lazy/static HTML load (A: lazy, B: SPA) | Low | Small (A) / Medium (B) | No template load at import; Option B: multi-device SPA |
| 3.1 | App state instead of globals | Medium | Large | Explicit state; testable |
| 3.2 | Extract JS to connect.js / jumps.js | Low | Medium | Cacheable JS; smaller HTML |
| 3.3 | Optional front-end build | Low | Small–Medium | Optional bundles |

---

## Notes

- **Order:** Do Phase 1 first (1.1 → 1.2 → 1.3). Then 2.1 and 2.2 can be done in parallel if desired; 2.3 is independent. Phase 3 can follow once 1–2 are stable.
- **2.3 Option B:** Use for multi-device (computers, pads, phones): one codebase, same URLs, responsive UI. Server routing (1.2) is unchanged; Option B adds client-side routing and a single SPA shell.
- **Testing:** After each step, run existing tests and manually smoke-test Connect, Jump Review, Device/Skater/Coach management, and session start/stop.
- **Backward compatibility:** For 2.2, keep old event_id URLs as redirects or thin wrappers until all clients are updated; then remove.
- **Documentation:** Update README or internal docs when adding routers, db package layout, or static asset layout.
