/* notes2latex-hybrid web app (no frameworks, no build step) */
"use strict";

const $ = (sel) => document.querySelector(sel);
const views = ["home", "jobs", "job", "settings"];
let currentJobId = null;
let eventSource = null;

function show(view) {
  for (const v of views) $("#view-" + v).classList.toggle("hidden", v !== view);
  document.querySelectorAll("nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.nav === view || (view === "job" && a.dataset.nav === "jobs"));
  });
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.status + " " + res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* noop */ }
    throw new Error(msg);
  }
  return res.json();
}

/* --------------------------------------------------------- native folders */
/* In the desktop window the page can open a real folder chooser through
   window.pywebview.api; in a plain browser tab there is no such thing, so we
   fall back to asking for the path. */
function inDesktopWindow() {
  return !!(window.pywebview && window.pywebview.api && window.pywebview.api.pick_folder);
}

async function pickFolder(current) {
  if (inDesktopWindow()) {
    try {
      const chosen = await window.pywebview.api.pick_folder(current || "");
      return chosen || "";
    } catch (e) { /* fall through to the prompt */ }
  }
  return window.prompt("Full path of the folder to save into:", current || "") || "";
}

/* ------------------------------------------------------------------ router */
function route() {
  if (eventSource) { eventSource.close(); eventSource = null; }
  const hash = location.hash || "#/";
  if (hash.startsWith("#/job/")) {
    currentJobId = hash.split("/")[2];
    show("job");
    loadJob(currentJobId, true);
  } else if (hash === "#/jobs") {
    show("jobs");
    loadJobs();
  } else if (hash === "#/settings") {
    show("settings");
    loadSettings();
  } else {
    show("home");
    loadEngineSummary();
  }
}
window.addEventListener("hashchange", route);

/* ------------------------------------------------------------------- home */
async function loadEngineSummary() {
  try {
    const [s, pre] = await Promise.all([api("/api/settings"), api("/api/preflight")]);
    $("#preflight-warning").classList.toggle("hidden", !!pre.latex_toolchain);
    const parts = [];
    parts.push(`<span class="badge accent">v${esc(pre.version)} engine: ${esc(s.engine)}</span>`);
    if (s.engine === "hybrid") {
      parts.push(`<span class="badge">primary: ${s.trocr_model_dir ? "trocr (local)" : "heuristic (local)"}</span>`);
    }
    if (s.vlm_base_url) {
      parts.push(`<span class="badge ${s.vlm_model ? "ok" : "bad"}">vlm: ${esc(s.vlm_model || "model missing")}</span>`);
      parts.push(`<span class="badge ${s.vlm_api_key_set || s.vlm_api_key ? "ok" : "warn"}">api key: ${s.vlm_api_key_set || s.vlm_api_key ? "saved" : "not set"}</span>`);
      if (pre.vlm_ok === true) {
        parts.push(`<span class="badge ok">endpoint: reachable</span>`);
      } else if (pre.vlm_ok === false) {
        parts.push(`<span class="badge bad" title="${esc(pre.vlm_error || "")}">endpoint: unreachable</span>`);
      }
    } else {
      parts.push(`<span class="badge warn">no VLM endpoint - fully offline</span>`);
    }
    $("#engine-summary").innerHTML = parts.join(" ");
  } catch (e) {
    $("#engine-summary").innerHTML = `<span class="badge bad">${esc(e.message)}</span>`;
  }
}

$("#upload-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const files = $("#file-input").files;
  const status = $("#upload-status");
  if (!files.length) { status.textContent = "choose at least one file"; status.className = "status err"; return; }
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  $("#upload-btn").disabled = true;
  status.textContent = "uploading..."; status.className = "status";
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    location.hash = "#/job/" + job.id;
  } catch (e) {
    status.textContent = e.message; status.className = "status err";
  } finally {
    $("#upload-btn").disabled = false;
  }
});

/* ------------------------------------------------------------------- jobs */
async function loadJobs() {
  const jobs = await api("/api/jobs");
  const el = $("#jobs-list");
  if (!jobs.length) { el.innerHTML = '<p class="muted">No jobs yet. Upload notes on the Convert tab.</p>'; return; }
  el.innerHTML = jobs.map((j) => {
    const cls = j.status === "done" ? "ok" : j.status === "error" ? "bad" : "warn";
    const when = new Date(j.created_at * 1000).toLocaleString();
    const res = j.result ? `${j.result.pages_ok}/${j.result.pages} pages` : "";
    return `<div class="job-row">
      <a href="#/job/${j.id}">${esc(j.id)}</a>
      <span class="muted">${esc((j.filenames || []).join(", ").slice(0, 60))}</span>
      <span class="muted">${when}</span>
      <span><span class="badge ${cls}">${esc(j.status)}</span> ${res}</span>
    </div>`;
  }).join("");
}

/* -------------------------------------------------------------------- job */
async function loadJob(id, live) {
  $("#job-title").textContent = id;
  $("#dl-tex").href = `/api/jobs/${id}/download/tex`;
  $("#dl-pdf").href = `/api/jobs/${id}/download/pdf`;
  $("#save-status").textContent = "";
  $("#save-status").className = "status";
  renderProgress([]);
  $("#job-pages").innerHTML = '<p class="muted">waiting for progress...</p>';

  const job = await api("/api/jobs/" + id);
  renderBadges(job);
  if (job.pages && job.pages.length) renderPages(id, job);
  if (live && (job.status === "pending" || job.status === "running")) {
    streamEvents(id);
  } else {
    const events = await api(`/api/jobs/${id}/events?after=0`);
    renderProgress(events.map((e) => e));
  }
}

function renderBadges(job) {
  const res = job.result;
  const badges = [`<span class="badge ${job.status === "done" ? "ok" : job.status === "error" ? "bad" : "warn"}">${esc(job.status)}</span>`];
  if (res) {
    badges.push(`<span class="badge ${res.ok ? "ok" : "bad"}">compile: ${res.ok ? "verified" : "failed"}</span>`);
    badges.push(`<span class="badge">${res.pages_ok}/${res.pages} pages</span>`);
  }
  if (job.error) badges.push(`<span class="badge bad" title="${esc(job.error)}">${esc(job.error.slice(0, 80))}</span>`);
  $("#job-badges").innerHTML = badges.join(" ");
  const running = job.status === "pending" || job.status === "running";
  const btn = $("#cancel-btn");
  btn.classList.toggle("hidden", !running);
  if (running && btn.dataset.cancelling !== job.id) btn.textContent = "Cancel job";
}

$("#cancel-btn").addEventListener("click", async (ev) => {
  const btn = ev.target;
  if (!currentJobId) return;
  btn.disabled = true; btn.textContent = "Cancelling...";
  btn.dataset.cancelling = currentJobId;
  try {
    await api(`/api/jobs/${currentJobId}/cancel`, { method: "POST" });
  } catch (e) {
    btn.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
});

$("#save-btn").addEventListener("click", async () => {
  if (!currentJobId) return;
  const status = $("#save-status");
  let start = "";
  try { start = (await api("/api/settings")).output_dir || ""; } catch (e) { /* noop */ }
  const dir = await pickFolder(start);
  if (!dir) return;
  const btn = $("#save-btn");
  btn.disabled = true;
  status.textContent = "saving..."; status.className = "status";
  try {
    const res = await api(`/api/jobs/${currentJobId}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir }),
    });
    status.textContent = `saved ${res.written.join(", ")} to ${res.dir}`;
    status.className = "status ok";
    if (inDesktopWindow() && window.pywebview.api.reveal) window.pywebview.api.reveal(res.dir);
  } catch (e) {
    status.textContent = e.message; status.className = "status err";
  } finally {
    btn.disabled = false;
  }
});

function streamEvents(id) {
  if (eventSource) eventSource.close();
  const es = new EventSource(`/api/jobs/${id}/events`);
  eventSource = es;
  es.onmessage = (m) => {
    let ev; try { ev = JSON.parse(m.data); } catch (e) { return; }
    appendProgress(ev);
    if (ev.type === "page_done" || ev.type === "job_done") refreshJob(id, es);
  };
  es.addEventListener("end", () => { es.close(); eventSource = null; refreshJob(id, es); });
  es.onerror = () => { es.close(); eventSource = null; };
}

let refreshTimer = null;
function refreshJob(id, es) {
  if (refreshTimer) clearTimeout(refreshTimer);
  refreshTimer = setTimeout(async () => {
    try {
      const job = await api("/api/jobs/" + id);
      renderBadges(job);
      if (job.pages && job.pages.length) renderPages(id, job);
      if (job.status === "done" || job.status === "error") { if (eventSource) { eventSource.close(); eventSource = null; } }
    } catch (e) { /* transient */ }
  }, es ? 300 : 0);
}

function renderProgress(events) {
  $("#job-progress").innerHTML = events.map(fmtEvent).join("");
  const box = $("#job-progress");
  box.scrollTop = box.scrollHeight;
}
function appendProgress(ev) {
  const box = $("#job-progress");
  box.insertAdjacentHTML("beforeend", fmtEvent(ev));
  box.scrollTop = box.scrollHeight;
}
function fmtEvent(ev) {
  const t = new Date((ev.ts || Date.now() / 1000) * 1000).toLocaleTimeString();
  let cls = "", msg = ev.type;
  if (ev.type === "compile_fail") { cls = "ev-bad"; msg = `page ${ev.page}: compile failed (${(ev.errors || []).join("; ").slice(0, 100)})`; }
  else if (ev.type === "page_done") { cls = ev.status === "failed" ? "ev-bad" : "ev-ok"; msg = `page ${ev.page}: ${ev.status} (${ev.attempts} attempt(s), ${ev.engine})`; }
  else if (ev.type === "job_done") { cls = ev.ok ? "ev-ok" : "ev-bad"; msg = `document ${ev.ok ? "compiled successfully" : "has unverified pages"} - ${ev.pages_ok}/${ev.pages} pages verified`; }
  else if (ev.type === "engine_error") { cls = "ev-warn"; msg = `engine error: ${String(ev.error).slice(0, 120)}`; }
  else if (ev.type === "ingested") { msg = `${ev.pages} page(s) ingested`; }
  else if (ev.type === "prefetch_start") { msg = `transcribing ${ev.total} pages in parallel (${ev.workers} workers)...`; }
  else if (ev.type === "prefetched") { cls = ev.ok ? "" : "ev-warn"; msg = `page ${ev.page}: prefetched (${ev.done}/${ev.total})${ev.ok ? "" : " - failed, will retry live: " + String(ev.error).slice(0, 80)}`; }
  else if (ev.type === "transcribed") { msg = `page ${ev.page}: transcribed (${ev.engine}, attempt ${ev.attempt})`; }
  else if (ev.type === "page_start") { msg = `page ${ev.page}: starting`; }
  else if (ev.type === "job_error") { cls = "ev-bad"; msg = `job error: ${String(ev.error).slice(0, 140)}`; }
  else if (ev.type === "job_aborted") { cls = "ev-bad"; msg = `job stopped early: ${String(ev.reason).slice(0, 200)}`; }
  else if (ev.type === "cancel_requested") { cls = "ev-warn"; msg = "cancel requested - stopping after the in-flight request is aborted"; }
  else if (ev.type === "job_cancelled") { cls = "ev-warn"; msg = "job cancelled - remaining pages not attempted"; }
  else if (ev.type === "saved") { cls = "ev-ok"; msg = `saved ${(ev.written || []).join(", ")} to ${ev.dir}`; }
  else if (ev.type === "save_failed") { cls = "ev-warn"; msg = `could not save to ${ev.dir}: ${String(ev.error).slice(0, 120)}`; }
  else if (ev.type === "autofix") { cls = ev.ok ? "ev-ok" : "ev-warn"; msg = `page ${ev.page}: auto-repaired (${(ev.changes || []).join("; ")})${ev.ok ? "" : " - still failing"}`; }
  return `<div class="${cls}">[${t}] ${esc(msg)}</div>`;
}

function renderPages(id, job) {
  $("#job-pages").innerHTML = job.pages.map((p) => `
    <div class="page-card">
      <div class="img-side"><img src="/api/jobs/${id}/pages/${p.index}/image" alt="page ${p.index}"></div>
      <div class="tex-side">
        <div class="page-meta">
          <span class="badge">page ${p.index}</span>
          <span class="badge ${p.status === "failed" ? "bad" : p.status === "fixed" ? "warn" : "ok"}">${esc(p.status)}</span>
          <span class="badge">${esc(p.engine)}</span>
          <span class="badge">${p.attempts} attempt(s)</span>
          <span class="spacer"></span>
          <button class="copy-btn" data-page="${p.index}">Copy LaTeX</button>
        </div>
        <textarea readonly data-latex="${p.index}">${esc(p.latex)}</textarea>
      </div>
    </div>`).join("");
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ta = document.querySelector(`textarea[data-latex="${btn.dataset.page}"]`);
      ta.select();
      navigator.clipboard.writeText(ta.value).then(() => {
        btn.textContent = "Copied";
        setTimeout(() => (btn.textContent = "Copy LaTeX"), 1200);
      });
    });
  });
}

/* --------------------------------------------------------------- settings */
async function loadSettings() {
  const s = await api("/api/settings");
  const form = $("#settings-form");
  for (const key of ["engine", "vlm_base_url", "vlm_model", "trocr_model_dir", "dpi", "max_retries", "context_lines", "vlm_timeout", "vlm_stall_timeout", "vlm_retries", "vlm_thinking", "vlm_parallel_workers", "doc_font_pt", "doc_paper", "doc_margin_in", "output_dir"]) {
    if (form.elements[key]) form.elements[key].value = s[key] ?? "";
  }
  form.elements.escalate_to_vlm.checked = !!s.escalate_to_vlm;
  form.elements.doc_landscape.checked = !!s.doc_landscape;
  form.elements.doc_two_column.checked = !!s.doc_two_column;
  form.elements.vlm_use_proxy.checked = !!s.vlm_use_proxy;
  form.elements.vlm_api_key.value = "";
  const keySaved = !!(s.vlm_api_key_set || s.vlm_api_key);
  form.elements.vlm_api_key.placeholder = keySaved ? "saved - leave blank to keep" : "not set";
  $("#key-status").innerHTML = keySaved
    ? `<span class="badge ok">API key: saved (${esc(s.vlm_api_key || "hidden")})</span>`
    : `<span class="badge bad">API key: not set</span>`;
}

$("#browse-output").addEventListener("click", async () => {
  const field = $("#settings-form").elements.output_dir;
  const dir = await pickFolder(field.value.trim());
  if (dir) field.value = dir;
});

$("#settings-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const payload = {
    engine: form.engine.value,
    vlm_base_url: form.vlm_base_url.value.trim(),
    vlm_model: form.vlm_model.value.trim(),
    trocr_model_dir: form.trocr_model_dir.value.trim(),
    dpi: form.dpi.value,
    max_retries: form.max_retries.value,
    context_lines: form.context_lines.value,
    vlm_timeout: form.vlm_timeout.value,
    vlm_stall_timeout: form.vlm_stall_timeout.value,
    vlm_retries: form.vlm_retries.value,
    vlm_thinking: form.vlm_thinking.value,
    vlm_parallel_workers: form.vlm_parallel_workers.value,
    doc_font_pt: form.doc_font_pt.value,
    doc_paper: form.doc_paper.value,
    doc_margin_in: form.doc_margin_in.value,
    doc_landscape: form.doc_landscape.checked,
    doc_two_column: form.doc_two_column.checked,
    output_dir: form.output_dir.value.trim(),
    escalate_to_vlm: form.escalate_to_vlm.checked,
    vlm_use_proxy: form.vlm_use_proxy.checked,
  };
  if (form.vlm_api_key.value) payload.vlm_api_key = form.vlm_api_key.value;
  const status = $("#settings-status");
  try {
    await api("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    status.textContent = "saved"; status.className = "status ok";
  } catch (e) {
    status.textContent = e.message; status.className = "status err";
  }
  loadEngineSummary();
});

route();
