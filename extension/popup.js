// popup.js — drives the HP+ Priority Checker popup UI

// ── Tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");

    if (tab.dataset.tab === "history") loadHistory();
  });
});

// ── Backend health check ──────────────────────────────────────────────────
async function checkBackend() {
  const dot = document.getElementById("statusDot");
  const statusEl = document.getElementById("backendStatus");
  const resp = await chrome.runtime.sendMessage({ type: "HEALTH" });
  if (resp && resp.ok) {
    dot.classList.add("ok");
    dot.title = "Backend online";
    if (statusEl) statusEl.textContent = `✓ Backend online — ${resp.data.projects} projects loaded`;
  } else {
    dot.classList.add("err");
    dot.title = "Backend offline";
    if (statusEl) statusEl.textContent = `✗ Backend offline — start app.py on port 5487`;
  }
}
checkBackend();

// ── Load last prediction result ──────────────────────────────────────────
async function loadResult() {
  const stored = await chrome.storage.local.get(["lastResult"]);
  const d = stored.lastResult;
  if (!d) return;
  renderResult(d);
}
loadResult();

// Auto-refresh popup when content.js writes a new result to storage
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.lastResult) {
    renderResult(changes.lastResult.newValue);
  }
});

function renderResult(d) {
  // Loading state written by content.js at check-start
  if (d && d.loading) {
    document.getElementById("no-result").style.display = "none";
    document.getElementById("error-box").style.display = "none";
    document.getElementById("has-result").style.display = "block";
    const card = document.getElementById("verdictCard");
    card.className = "verdict-card none";
    document.getElementById("verdictIcon").textContent  = "⏳";
    document.getElementById("verdictLabel").textContent = "Analyzing…";
    document.getElementById("verdictProb").textContent  = "Fetching prediction";
    document.getElementById("metaChips").innerHTML = "";
    document.getElementById("probFill").style.width = "0%";
    document.getElementById("probPct").textContent  = "—";
    const exBox = document.getElementById("explanationBox");
    exBox.className = "explanation loading";
    exBox.textContent = "Running model and generating explanation…";
    renderFeatures([]);
    return;
  }

  // Error state
  if (d && d.error) {
    document.getElementById("no-result").style.display = "none";
    document.getElementById("has-result").style.display = "none";
    const errBox = document.getElementById("error-box");
    errBox.style.display = "block";
    errBox.textContent = d.error;
    return;
  }

  if (!d || !d.verdict) return;

  document.getElementById("no-result").style.display = "none";
  document.getElementById("error-box").style.display = "none";
  document.getElementById("has-result").style.display = "block";

  const isHp  = d.verdict === "HP+";
  const card  = document.getElementById("verdictCard");
  card.className = `verdict-card ${isHp ? "hp" : "ok"}`;
  document.getElementById("verdictIcon").textContent  = isHp ? "🔴" : "🟢";
  document.getElementById("verdictLabel").textContent = d.verdict;
  document.getElementById("verdictProb").textContent  =
    `${Math.round(d.probability * 100)}% HP+ probability`;

  // Probability bar
  const pct = Math.round(d.probability * 100);
  document.getElementById("probFill").style.width = `${pct}%`;
  document.getElementById("probPct").textContent  = `${pct}%`;
  const thrPct = Math.round((d.threshold_used || 0.5) * 100);
  document.getElementById("thrLine").style.left   = `${thrPct}%`;

  // Meta chips (training-set AUC/MCC are deliberately not shown: they are
  // in-sample and overstate real performance)
  const chips = document.getElementById("metaChips");
  chips.innerHTML = [
    d.matched_category,
    `thr: ${thrPct}% (${d.threshold_type})`,
  ].filter(Boolean).map(t =>
    `<span class="meta-chip">${esc(t)}</span>`
  ).join("");

  // LLM explanation — parse structured markdown sections
  const exBox = document.getElementById("explanationBox");
  if (d.llm_explanation && !d.llm_explanation.startsWith("[LLM")) {
    exBox.className = "explanation";
    exBox.innerHTML = renderStructuredLlm(d.llm_explanation);
  } else {
    exBox.className = "explanation loading";
    exBox.textContent = d.llm_explanation || "No explanation available.";
  }

  // Risk dashboard
  renderFeatures(d.top_features || []);

  // Sync slider to current threshold
  const sliderEl = document.getElementById("thrSlider");
  sliderEl.value = thrPct;
  document.getElementById("thrVal").textContent = `${thrPct}%`;
}

/**
 * Parse the structured LLM markdown into HTML sections.
 *
 * A section header is **Title**: at the VERY START of a line.
 * Bullet items like "- **Sub-label**: text" are rendered as <li> entries,
 * NOT promoted to section headers (the old regex matched anywhere in the text).
 */
function renderStructuredLlm(text) {
  const lines = text.split("\n");
  // Matches **Title**: only when the line starts with it (no leading - or whitespace before **)
  const headerRe = /^\*\*([^*]+)\*\*\s*:\s*(.*)/;
  const bulletRe  = /^[-•*]\s+(.*)/;

  const sections = [];
  let cur = null;

  for (const line of lines) {
    const hm = line.trim().match(headerRe);
    // Treat as section header only if the trimmed line starts with ** (not a bullet)
    if (hm && !bulletRe.test(line.trim())) {
      cur = { title: hm[1].trim(), trailer: hm[2].trim(), items: [] };
      sections.push(cur);
    } else if (cur !== null) {
      cur.items.push(line);
    }
  }

  if (sections.length === 0) {
    return `<div class="llm-section"><div class="llm-section-body">${esc(text)}</div></div>`;
  }

  let html = "";
  for (const { title, trailer, items } of sections) {
    const nonEmpty = items.map(l => l.trim()).filter(Boolean);
    const hasBullets = nonEmpty.some(l => bulletRe.test(l));

    let bodyHtml = "";
    if (hasBullets) {
      let lead = trailer ? `<span>${esc(trailer)}</span> ` : "";
      let listItems = "";
      for (const line of nonEmpty) {
        const bm = line.match(bulletRe);
        if (bm) {
          listItems += `<li>${formatBulletText(bm[1])}</li>`;
        } else {
          lead += `<span>${esc(line)}</span> `;
        }
      }
      bodyHtml = lead + `<ul class="llm-bullets">${listItems}</ul>`;
    } else {
      const all = [trailer, ...nonEmpty].filter(Boolean);
      bodyHtml = esc(all.join(" "));
    }

    html += `<div class="llm-section"><div class="llm-section-title">${esc(title)}</div><div class="llm-section-body">${bodyHtml}</div></div>`;
  }
  return html;
}

// Render a bullet's text: converts "**Sub-label**: rest" to <strong>Sub-label</strong>: rest
function formatBulletText(raw) {
  const m = raw.match(/^\*\*([^*]+)\*\*:\s*(.*)/s);
  if (m) return `<strong>${esc(m[1])}</strong>: ${esc(m[2])}`;
  return esc(raw);
}

function renderFeatures(features) {
  const list = document.getElementById("featureList");
  if (!features.length) {
    list.innerHTML = `<div style="color:#aaa;font-size:12px">No data</div>`;
    return;
  }
  const maxImp = Math.max(...features.map(f => f.importance), 0.001);
  list.innerHTML = features.map(f => {
    const pct = Math.round((f.importance / maxImp) * 100);
    return `
      <div class="feature-row">
        <span class="feature-name" title="${esc(f.name)}">${esc(f.name)}</span>
        <div class="feature-bar-track">
          <div class="feature-bar-fill" style="width:${pct}%"></div>
        </div>
        <span class="feature-imp">${f.importance.toFixed(3)}</span>
      </div>
    `;
  }).join("");
}

// ── History tab ────────────────────────────────────────────────────────────
async function loadHistory() {
  const resp = await chrome.runtime.sendMessage({ type: "GET_HISTORY", limit: 30 });
  const tbody = document.getElementById("historyBody");
  if (!resp || !resp.ok || !resp.data.length) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:#aaa;padding:12px">No history yet</td></tr>`;
    return;
  }
  tbody.innerHTML = resp.data.map(row => {
    const proj  = (row.project || "").replace("openstack/", "");
    const badge = row.verdict === "HP+"
      ? `<span class="badge-hp">HP+</span>`
      : `<span class="badge-norm">Normal</span>`;
    const prob  = row.probability != null ? `${Math.round(row.probability * 100)}%` : "—";
    const change = `<a href="https://review.opendev.org/c/${esc(row.project)}/+/${esc(row.change_id)}"
                       target="_blank" style="color:#1565C0;text-decoration:none">
                       #${esc(row.change_id)}</a>`;
    return `<tr><td>${change}</td><td>${esc(proj)}</td><td>${badge}</td><td>${esc(prob)}</td></tr>`;
  }).join("");
}

// ── Settings / Threshold slider ───────────────────────────────────────────
const slider = document.getElementById("thrSlider");
const thrVal = document.getElementById("thrVal");
slider.addEventListener("input", () => {
  thrVal.textContent = `${slider.value}%`;
  document.querySelector('input[name=thrType][value=custom]').checked = true;
});

document.querySelectorAll('input[name=thrType]').forEach(r => {
  r.addEventListener("change", async () => {
    if (r.value !== "custom") {
      // Load model threshold for this type
      const stored = await chrome.storage.local.get(["lastResult"]);
      const d = stored.lastResult;
      if (d && d.threshold_used) {
        slider.value = Math.round(d.threshold_used * 100);
        thrVal.textContent = `${slider.value}%`;
      }
    }
  });
});

document.getElementById("applyThr").addEventListener("click", async () => {
  const thrType = document.querySelector('input[name=thrType]:checked')?.value || "gmean";
  const customVal = parseInt(slider.value, 10) / 100;

  await chrome.storage.local.set({
    thresholdType:     thrType,
    thresholdOverride: thrType === "custom" ? customVal : null,
  });

  // Re-trigger prediction on active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) {
    chrome.tabs.sendMessage(tab.id, { type: "RERUN" });
  }
});

// ── Helpers ────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
