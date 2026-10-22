// Service worker — bridges HTTPS Gerrit page → HTTP localhost backend.
// Extension service workers can fetch http://localhost even from HTTPS pages.

const BACKEND = "http://127.0.0.1:5487";

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "PREDICT") {
    handlePredict(msg.payload).then(sendResponse);
    return true;
  }
  if (msg.type === "PREDICT_BATCH") {
    handlePredictBatch(msg.payload).then(sendResponse);
    return true;
  }
  if (msg.type === "GET_HISTORY") {
    handleHistory(msg.limit || 20).then(sendResponse);
    return true;
  }
  if (msg.type === "HEALTH") {
    handleHealth().then(sendResponse);
    return true;
  }
});

async function handlePredict(payload) {
  try {
    const r = await fetch(`${BACKEND}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    return { ok: r.ok, status: r.status, data };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

async function handlePredictBatch(changes) {
  try {
    const r = await fetch(`${BACKEND}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ changes }),
    });
    const data = await r.json();
    return { ok: r.ok, status: r.status, data };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

async function handleHistory(limit) {
  try {
    const r = await fetch(`${BACKEND}/history?limit=${limit}`);
    const data = await r.json();
    return { ok: r.ok, data };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

async function handleHealth() {
  try {
    const r = await fetch(`${BACKEND}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await r.json();
    return { ok: r.ok, data };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}
