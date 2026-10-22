// Content script — injects HP+ verdict on both detail pages and list pages.
// Runs on https://review.opendev.org/*

const BANNER_ID   = "hp-checker-banner";
const BADGE_ATTR  = "data-hp-checked";
const SUPPORTED   = ["openstack/nova", "openstack/neutron", "openstack/cinder",
                      "openstack/glance", "openstack/designate"];

// ── URL parsing ────────────────────────────────────────────────────────────
// Detail page: /c/openstack/nova/+/12345
function parseGerritUrl(url) {
  const m = url.match(/\/c\/(openstack\/[^/]+)\/\+\/(\d+)/);
  if (!m) return null;
  return { project: m[1], changeNum: m[2] };
}

// ── HTML escaping ──────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ═══════════════════════════════════════════════════════════════════════════
// DETAIL PAGE — fixed top banner (unchanged from original)
// ═══════════════════════════════════════════════════════════════════════════

function injectBanner(state) {
  removeBanner();
  const banner = document.createElement("div");
  banner.id = BANNER_ID;
  banner.style.cssText = `
    position: fixed; top: 0; left: 0; right: 0; z-index: 999999;
    font-family: 'Roboto', sans-serif; font-size: 13px;
    padding: 6px 16px; display: flex; align-items: center; gap: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    ${state.loading ? "background:#1565C0; color:#fff;" :
      state.error   ? "background:#B71C1C; color:#fff;" :
      state.verdict === "HP+" ? "background:#C62828; color:#fff;" :
                                "background:#1B5E20; color:#fff;"}
  `;
  if (state.loading) {
    banner.innerHTML = `
      <span style="font-weight:600">⏳ HP+ Checker</span>
      <span>Analysing change…</span>
    `;
  } else if (state.error) {
    banner.innerHTML = `
      <span style="font-weight:600">⚠ HP+ Checker</span>
      <span>${escHtml(state.error)}</span>
      <button id="hp-close" style="margin-left:auto;background:transparent;border:1px solid #fff;color:#fff;padding:2px 8px;cursor:pointer;border-radius:3px">✕</button>
    `;
  } else {
    const pct  = Math.round(state.probability * 100);
    const icon = state.verdict === "HP+" ? "🔴" : "🟢";
    banner.innerHTML = `
      <span style="font-size:16px">${icon}</span>
      <span style="font-weight:700;font-size:14px">${escHtml(state.verdict)}</span>
      <span style="opacity:0.85">${pct}% HP+ probability</span>
      <span style="opacity:0.7;font-size:11px">(thr ${(state.threshold_used * 100).toFixed(0)}% · ${escHtml(state.threshold_type)})</span>
      <span style="opacity:0.75;margin-left:4px">${escHtml(state.matched_category)}</span>
      <button id="hp-close" style="margin-left:auto;background:transparent;border:1px solid rgba(255,255,255,0.6);color:#fff;padding:2px 8px;cursor:pointer;border-radius:3px">✕</button>
    `;
  }
  document.body.prepend(banner);
  document.getElementById("hp-close")?.addEventListener("click", removeBanner);
}

function removeBanner() {
  document.getElementById(BANNER_ID)?.remove();
}

let _lastChangeKey = "";

async function runCheck() {
  const info = parseGerritUrl(location.href);
  if (!info) return;
  if (!SUPPORTED.includes(info.project)) return;

  const key = `${info.project}:${info.changeNum}`;
  if (key === _lastChangeKey) return;
  _lastChangeKey = key;

  injectBanner({ loading: true });

  const stored = await chrome.storage.local.get(["thresholdType", "thresholdOverride"]);
  const thresholdType     = stored.thresholdType     || "gmean";
  const thresholdOverride = stored.thresholdOverride ?? null;

  const payload = {
    change_id:      info.changeNum,
    project:        info.project,
    threshold_type: thresholdType,
    ...(thresholdOverride !== null ? { threshold_override: thresholdOverride } : {}),
  };

  // Write loading state immediately so popup opens in sync with banner
  await chrome.storage.local.set({ lastResult: { loading: true, changeKey: key } });

  const resp = await chrome.runtime.sendMessage({ type: "PREDICT", payload });
  if (!resp || !resp.ok) {
    const errState = { error: resp?.error || `Backend error (${resp?.status})`, changeKey: key };
    await chrome.storage.local.set({ lastResult: errState });
    injectBanner({ error: errState.error });
    return;
  }
  const d = resp.data;
  if (d.error) {
    const errState = { error: d.error, changeKey: key };
    await chrome.storage.local.set({ lastResult: errState });
    injectBanner({ error: d.error });
    return;
  }
  // Write result BEFORE showing banner so popup always reflects the same state
  await chrome.storage.local.set({ lastResult: { ...d, changeKey: key } });
  injectBanner(d);
}

// ═══════════════════════════════════════════════════════════════════════════
// LIST PAGE — compact inline badges
// ═══════════════════════════════════════════════════════════════════════════

// Recursively walk all shadow roots to find elements matching tagName.
// PolyGerrit renders gr-change-list-item inside gr-change-list-section's
// shadow DOM, which is itself nested several shadow roots deep from <gr-app>.
function deepQueryShadow(tagName) {
  const results = [];
  function walk(root) {
    try {
      root.querySelectorAll(tagName).forEach(el => results.push(el));
      root.querySelectorAll("*").forEach(el => {
        if (el.shadowRoot) walk(el.shadowRoot);
      });
    } catch (_) {}
  }
  walk(document);
  return results;
}

// Inspect every gr-change-list-item in the shadow tree.
// Returns:
//   batch  — array of {change_id, project} to send to /predict_batch
//   rowMap — "project:change_id" → <td class="cell subject"> for badge injection
function collectListItems() {
  const items  = deepQueryShadow("gr-change-list-item");
  const batch  = [];
  const rowMap = {};

  for (const item of items) {
    const sr = item.shadowRoot;
    if (!sr) continue;

    // The subject cell — confirmed from PolyGerrit source:
    //   renderCellSubject() returns <td class="cell subject"><a href=URL>…
    const td = sr.querySelector("td.cell.subject");
    if (!td) continue;

    // Skip rows already badged
    if (td.getAttribute(BADGE_ATTR)) continue;

    const a    = td.querySelector("a");
    const href = a?.getAttribute("href") || "";
    const m    = href.match(/\/c\/(openstack\/[^/]+)\/\+\/(\d+)/);
    if (!m) continue;

    const [, project, changeNum] = m;
    if (!SUPPORTED.includes(project)) continue;

    batch.push({ change_id: changeNum, project });
    rowMap[`${project}:${changeNum}`] = td;
  }

  return { batch, rowMap };
}

function injectBadge(td, result) {
  // Remove any stale badge first (shouldn't happen thanks to BADGE_ATTR guard,
  // but defensive just in case the attribute was cleared by a re-render)
  td.querySelector(".hp-badge")?.remove();

  const isHp  = result.verdict === "HP+";
  const badge = document.createElement("span");
  badge.className  = "hp-badge";
  badge.style.cssText = [
    "display:inline-flex", "align-items:center", "gap:3px",
    "padding:1px 7px", "border-radius:10px", "font-size:11px", "font-weight:600",
    "margin-left:8px", "vertical-align:middle", "white-space:nowrap",
    "line-height:16px",
    `background:${isHp ? "#FFEBEE" : "#E8F5E9"}`,
    `color:${isHp ? "#C62828" : "#1B5E20"}`,
    `border:1px solid ${isHp ? "#FFCDD2" : "#C8E6C9"}`,
  ].join(";");
  badge.textContent = isHp ? "🔴 HP+" : "🟢 Normal";
  badge.title = `${Math.round(result.probability * 100)}% HP+ probability · ${result.matched_category}`;

  td.setAttribute(BADGE_ATTR, "1");
  // Append after the subject <a> link, inside the <td>
  td.appendChild(badge);
}

let _listDebounceTimer = null;

async function runListCheck() {
  const { batch, rowMap } = collectListItems();
  if (batch.length === 0) return;

  // One batch request for all unbadged rows on this page
  let resp;
  try {
    resp = await chrome.runtime.sendMessage({ type: "PREDICT_BATCH", payload: batch });
  } catch (_) {
    return; // backend offline — fail silently on list pages
  }
  if (!resp || !resp.ok || !Array.isArray(resp.data)) return;

  for (const result of resp.data) {
    const key = `${result.project}:${result.change_id}`;
    const td  = rowMap[key];
    if (td) injectBadge(td, result);
  }
}

function scheduleListCheck() {
  clearTimeout(_listDebounceTimer);
  _listDebounceTimer = setTimeout(runListCheck, 400);
}

// ═══════════════════════════════════════════════════════════════════════════
// SPA NAVIGATION — shared MutationObserver
// ═══════════════════════════════════════════════════════════════════════════

let _prevUrl = location.href;

const _observer = new MutationObserver(() => {
  if (location.href !== _prevUrl) {
    // URL changed → SPA navigation to a new page
    _prevUrl      = location.href;
    _lastChangeKey = "";
    runCheck();           // detail page banner (no-ops if not a change page)
    scheduleListCheck();  // list page badges (no-ops if no gr-change-list-item found)
  } else {
    // DOM mutated without URL change — pagination, tab switch, or row updates
    // on a list page; skip on detail pages where mutations are frequent
    if (!parseGerritUrl(location.href)) {
      scheduleListCheck();
    }
  }
});

// Wait for <gr-app> (PolyGerrit's root Web Component) before starting
function waitForGerrit() {
  const grApp = document.querySelector("gr-app");
  if (grApp) {
    _observer.observe(document.body, { childList: true, subtree: true });
    runCheck();          // in case we landed directly on a detail page
    scheduleListCheck(); // in case we landed directly on a list page
  } else {
    setTimeout(waitForGerrit, 500);
  }
}

waitForGerrit();

// Listen for re-run request from popup (threshold changed)
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "RERUN") {
    _lastChangeKey = "";
    runCheck();
  }
});
