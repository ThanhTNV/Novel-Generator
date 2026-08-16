/* ============================================================
   Novel Generator — frontend
   ------------------------------------------------------------
   One rule runs through this file: every request that touches a
   story carries the current novel id. There is no ambient scope
   on the server, so a request without one silently means "the
   default workspace" — which is exactly the leak the separate
   folders exist to prevent.
   ============================================================ */

const $ = (id) => document.getElementById(id);
const state = {
  novel: null,          // slug of the active workspace
  novels: [],
  chatHistory: [],
  streaming: false,
  contextFile: null,    // filename currently open in the World editor
};

const LAST_NOVEL_KEY = "novelgen.novel";
const THEME_KEY = "novelgen.theme";

/* ---------- panel metadata ---------- */
const PANELS = {
  compose:    { title: "Compose",    hint: "Draft a chapter grounded in this novel's memory." },
  manuscript: { title: "Manuscript", hint: "Write by hand. Saving indexes it for later chapters." },
  chapters:   { title: "Chapters",   hint: "Everything you've saved for this novel." },
  world:      { title: "World",      hint: "The bible this novel's characters and places come from." },
  chat:       { title: "Ask",        hint: "Questions answered only from this novel's memory." },
  memory:     { title: "Memory",     hint: "What Zero-Mem has indexed, and what it retrieves." },
  settings:   { title: "Settings",   hint: "Providers, rules and skills in effect." },
};

/* ============================================================
   HTTP
   ============================================================ */

function withNovel(path) {
  if (!state.novel) return path;
  return path + (path.includes("?") ? "&" : "?") + "novel=" + encodeURIComponent(state.novel);
}

async function readError(res) {
  try {
    const data = await res.json();
    return data.detail || JSON.stringify(data);
  } catch {
    return (await res.text()) || `${res.status} ${res.statusText}`;
  }
}

async function api(method, path, body) {
  const opts = { method };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    // Scope goes in the body for writes, the query string for reads.
    opts.body = JSON.stringify(state.novel ? { novel: state.novel, ...body } : body);
  }
  const res = await fetch(body !== undefined ? path : withNovel(path), opts);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

const apiGet = (p) => api("GET", p);
const apiPost = (p, b) => api("POST", p, b || {});
const apiPut = (p, b) => api("PUT", p, b || {});
const apiDelete = (p) => api("DELETE", p);

/* ============================================================
   Chrome: toasts, busy, dialog
   ============================================================ */

function toast(message, type = "info") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  $("toastContainer").appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 200);
  }, type === "error" ? 6000 : 3500);
}

function busy(on, text = "Working…") {
  $("loadingText").textContent = text;
  $("loadingOverlay").hidden = !on;
}

function dialog({ title, body, confirmLabel = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    const overlay = $("dialog");
    const ok = $("dialogOk");
    const cancel = $("dialogCancel");

    $("dialogTitle").textContent = title;
    $("dialogBody").innerHTML = body;
    ok.textContent = confirmLabel;
    ok.className = "btn " + (danger ? "btn-danger" : "btn-accent");
    overlay.hidden = false;

    const firstInput = overlay.querySelector("input, textarea");
    if (firstInput) setTimeout(() => firstInput.focus(), 30);

    function done(result) {
      overlay.hidden = true;
      ok.removeEventListener("click", onOk);
      cancel.removeEventListener("click", onCancel);
      overlay.removeEventListener("click", onBackdrop);
      document.removeEventListener("keydown", onKey);
      resolve(result);
    }
    const onOk = () => done(true);
    const onCancel = () => done(false);
    const onBackdrop = (e) => { if (e.target === overlay) done(false); };
    const onKey = (e) => {
      if (e.key === "Escape") done(false);
      if (e.key === "Enter" && e.target.tagName !== "TEXTAREA") { e.preventDefault(); done(true); }
    };

    ok.addEventListener("click", onOk);
    cancel.addEventListener("click", onCancel);
    overlay.addEventListener("click", onBackdrop);
    document.addEventListener("keydown", onKey);
  });
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text == null ? "" : String(text);
  return d.innerHTML;
}

function emptyState(icon, heading, detail) {
  return `<div class="empty">${icon}<strong>${escapeHtml(heading)}</strong><p>${escapeHtml(detail)}</p></div>`;
}

const ICON_BOOK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 5.5A2.5 2.5 0 0 1 5.5 3H11v18H5.5A2.5 2.5 0 0 1 3 18.5z"/><path d="M21 5.5A2.5 2.5 0 0 0 18.5 3H13v18h5.5a2.5 2.5 0 0 0 2.5-2.5z"/></svg>`;
const ICON_GLOBE = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 0 1 0 18a15 15 0 0 1 0-18z"/></svg>`;
const ICON_SEARCH = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`;
const ICON_TRASH = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>`;

/* ============================================================
   Theme
   ============================================================ */

function applyTheme(mode) {
  if (mode === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", mode);
  localStorage.setItem(THEME_KEY, mode);
}

function currentTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored) return stored;
  return "system";
}

$("btnTheme").addEventListener("click", () => {
  // Three states, but the button cycles through the two the user can see;
  // "system" is the starting point and reachable again by clearing storage.
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const now = currentTheme();
  const resolved = now === "system" ? (dark ? "dark" : "light") : now;
  applyTheme(resolved === "dark" ? "light" : "dark");
});

applyTheme(currentTheme());

/* ============================================================
   Novels
   ============================================================ */

function currentNovel() {
  return state.novels.find((n) => n.slug === state.novel);
}

function renderNovelList() {
  const list = $("novelList");
  list.innerHTML = state.novels.map((n) => `
    <div class="novel-row${n.slug === state.novel ? " is-current" : ""}" data-slug="${escapeHtml(n.slug)}" role="option" tabindex="0">
      <div class="novel-row-body">
        <div class="novel-row-name">${escapeHtml(n.title)}</div>
        <div class="novel-row-meta">${n.chapters} chapter${n.chapters === 1 ? "" : "s"} · ${n.context_files} context file${n.context_files === 1 ? "" : "s"}</div>
      </div>
      ${n.slug === "default" ? "" : `<button class="novel-row-del" data-del="${escapeHtml(n.slug)}" title="Delete novel" type="button">${ICON_TRASH}</button>`}
    </div>`).join("");

  list.querySelectorAll(".novel-row").forEach((row) => {
    const pick = (e) => {
      if (e.target.closest("[data-del]")) return;
      selectNovel(row.dataset.slug);
      closePopover();
    };
    row.addEventListener("click", pick);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(e); } });
  });

  list.querySelectorAll("[data-del]").forEach((btn) => {
    btn.addEventListener("click", (e) => { e.stopPropagation(); deleteNovel(btn.dataset.del); });
  });
}

function reflectNovel() {
  const n = currentNovel();
  const name = n ? n.title : "—";
  $("currentNovelName").textContent = name;
  $("topbarScopeName").textContent = name;
  $("dangerNovelName").textContent = name;
  if (n) {
    $("navChapterCount").textContent = n.chapters;
    $("navChapterCount").hidden = !n.chapters;
    $("navWorldCount").textContent = n.context_files;
    $("navWorldCount").hidden = !n.context_files;
  }
  renderNovelList();
}

async function loadNovels(preferred) {
  const data = await apiGet("/api/novels");
  state.novels = data.novels;
  const stored = preferred || localStorage.getItem(LAST_NOVEL_KEY);
  const exists = state.novels.some((n) => n.slug === stored);
  state.novel = exists ? stored : (state.novels[0] && state.novels[0].slug) || data.default;
  localStorage.setItem(LAST_NOVEL_KEY, state.novel);
  reflectNovel();
}

async function selectNovel(slug) {
  if (slug === state.novel) return;
  state.novel = slug;
  localStorage.setItem(LAST_NOVEL_KEY, slug);
  reflectNovel();
  // Everything on screen belongs to the previous novel — clear it rather than
  // leaving one book's draft above another book's chapter list.
  state.chatHistory = [];
  state.contextFile = null;
  resetChat();
  $("contextEditor").value = "";
  $("contextFilename").value = "";
  $("btnDeleteContext").hidden = true;
  await refreshAll();
  toast(`Switched to “${currentNovel()?.title ?? slug}”`, "info");
}

function openPopover() {
  $("novelPopover").hidden = false;
  $("btnSwitcher").setAttribute("aria-expanded", "true");
}
function closePopover() {
  $("novelPopover").hidden = true;
  $("btnSwitcher").setAttribute("aria-expanded", "false");
}

$("btnSwitcher").addEventListener("click", (e) => {
  e.stopPropagation();
  $("novelPopover").hidden ? openPopover() : closePopover();
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".switcher")) closePopover();
});

$("btnNewNovel").addEventListener("click", async () => {
  closePopover();
  const ok = await dialog({
    title: "New novel",
    confirmLabel: "Create",
    body: `<p>A new novel gets its own context files, chapters and memory. Nothing is shared with your other novels.</p>
           <div class="field"><label for="dlgTitle">Title</label><input type="text" id="dlgTitle" placeholder="Đồ Lục Ký Sự"></div>
           <div class="field"><label for="dlgDesc">Description <span class="opt">optional</span></label><input type="text" id="dlgDesc" placeholder="Urban fantasy, Vietnamese"></div>`,
  });
  if (!ok) return;

  const title = ($("dlgTitle")?.value || "").trim();
  if (!title) { toast("A novel needs a title.", "error"); return; }

  try {
    // Creating a novel is the one write that must NOT carry a novel scope:
    // api() would inject the current one into the body and the server would
    // read it as part of the new novel's definition.
    const res = await fetch("/api/novels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, description: ($("dlgDesc")?.value || "").trim() }),
    });
    if (!res.ok) throw new Error(await readError(res));
    const created = await res.json();
    await loadNovels(created.slug);
    await refreshAll();
    switchPanel("world");
    toast(`Created “${created.title}”. Add its world bible to get started.`, "success");
  } catch (err) {
    toast(err.message, "error");
  }
});

async function deleteNovel(slug) {
  const n = state.novels.find((x) => x.slug === slug);
  const ok = await dialog({
    title: "Delete this novel?",
    confirmLabel: "Delete forever",
    danger: true,
    body: `<p>This permanently removes the folder, every chapter file, the world bible and the memory store.</p>
           <div class="dialog-detail"><b>${escapeHtml(n?.title || slug)}</b>
           <span>${n?.chapters ?? 0} chapters · ${n?.context_files ?? 0} context files</span></div>
           <p>This cannot be undone.</p>`,
  });
  if (!ok) return;

  try {
    const res = await fetch(`/api/novels/${encodeURIComponent(slug)}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await readError(res));
    if (state.novel === slug) state.novel = null;
    await loadNovels();
    await refreshAll();
    toast("Novel deleted.", "success");
  } catch (err) {
    toast(err.message, "error");
  }
}

/* ============================================================
   Navigation
   ============================================================ */

function switchPanel(name) {
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("is-active", b.dataset.panel === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("is-active", p.id === `panel-${name}`));

  const meta = PANELS[name];
  $("panelTitle").textContent = meta.title;
  $("panelHint").textContent = meta.hint;
  $("topbarScope").hidden = (name === "settings");
  $("rail").classList.remove("is-open");

  if (name === "manuscript") autoDetectNextChapter();
  if (name === "chapters") loadChapters();
  if (name === "world") loadContextFiles();
  if (name === "memory") loadMemory();
  if (name === "settings") loadSettings();
}

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => switchPanel(btn.dataset.panel));
});

$("btnRailToggle").addEventListener("click", () => $("rail").classList.toggle("is-open"));

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    $("novelPopover").hidden ? openPopover() : closePopover();
  }
});

/* ============================================================
   Streaming helper
   ============================================================ */

/**
 * Read an SSE response, calling onToken for each chunk.
 * The server emits {"error": ...} when a generation fails mid-stream; without
 * surfacing it the UI would just stop, leaving a half-written chapter and no
 * explanation.
 */
async function readStream(res, { onToken, onSources }) {
  if (!res.ok) throw new Error(await readError(res));
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let failed = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6);
      if (raw === "[DONE]") continue;
      let parsed;
      try { parsed = JSON.parse(raw); } catch { continue; }
      if (parsed.error) { failed = parsed.error; continue; }
      if (parsed.sources && onSources) onSources(parsed.sources);
      if (parsed.token && onToken) onToken(parsed.token);
    }
  }
  if (failed) throw new Error(failed);
}

/* ============================================================
   Compose
   ============================================================ */

const draftEditor = $("draftEditor");

function countWords(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function updateDraftCount() {
  const n = countWords(draftEditor.innerText || "");
  $("wordCount").textContent = `${n} word${n === 1 ? "" : "s"}`;
}

draftEditor.addEventListener("input", updateDraftCount);

$("temperature").addEventListener("input", (e) => { $("tempValue").textContent = e.target.value; });

function parseList(value) {
  return value.split(",").map((s) => s.trim()).filter(Boolean);
}

function generatePayload(stream) {
  return {
    chapter_instructions: $("chapterInstructions").value.trim(),
    story_summary: $("storySummary").value.trim(),
    characters: parseList($("characters").value),
    locations: parseList($("locations").value),
    target_words: parseInt($("targetWords").value, 10) || 2000,
    temperature: parseFloat($("temperature").value),
    // max_tokens omitted on purpose: the server sizes the output budget from
    // target_words. Vietnamese runs ~2.4 tokens/word, so a fixed 4096 cut
    // every default-length chapter off mid-sentence.
    provider: $("provider").value || null,
    model: $("model").value || null,
    stream,
  };
}

function revealDraftActions() {
  $("feedbackSection").hidden = false;
  $("finalizeSection").hidden = false;
}

function setComposeBusy(on) {
  state.streaming = on;
  ["btnGenerate", "btnStream"].forEach((id) => { $(id).disabled = on; });
  $("btnStream").textContent = on ? "Writing…" : "";
  if (!on) {
    $("btnStream").innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m13 2-9 12h7l-1 8 9-12h-7z"/></svg>Write chapter`;
  }
}

$("btnGenerate").addEventListener("click", async () => {
  const payload = generatePayload(false);
  if (!payload.chapter_instructions) { toast("Describe what happens in this chapter first.", "error"); return; }
  busy(true, "Writing the chapter…");
  try {
    const data = await apiPost("/api/generate", payload);
    draftEditor.innerText = data.content;
    updateDraftCount();
    revealDraftActions();
    toast("Draft ready.", "success");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    busy(false);
  }
});

$("btnStream").addEventListener("click", async () => {
  const payload = generatePayload(true);
  if (!payload.chapter_instructions) { toast("Describe what happens in this chapter first.", "error"); return; }

  draftEditor.innerText = "";
  setComposeBusy(true);
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ novel: state.novel, ...payload }),
    });
    await readStream(res, {
      onToken: (t) => {
        draftEditor.innerText += t;
        draftEditor.scrollTop = draftEditor.scrollHeight;
        updateDraftCount();
      },
    });
    revealDraftActions();
    toast("Draft ready.", "success");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    setComposeBusy(false);
  }
});

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    const active = document.querySelector(".panel.is-active");
    if (active && active.id === "panel-compose" && !state.streaming) {
      e.preventDefault();
      $("btnStream").click();
    }
  }
});

/* ---------- revise ---------- */

async function doRevise(stream) {
  const draft = draftEditor.innerText;
  const feedback = $("feedbackText").value.trim();
  if (!draft) { toast("There's no draft to revise.", "error"); return; }
  if (!feedback) { toast("Say what should change.", "error"); return; }

  const payload = {
    draft, feedback, temperature: 0.5,
    provider: $("provider").value || null,
    model: $("model").value || null,
    stream,
  };

  const buttons = ["btnRevise", "btnReviseStream"];
  buttons.forEach((id) => { $(id).disabled = true; });

  try {
    if (stream) {
      draftEditor.innerText = "";
      const res = await fetch("/api/revise", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ novel: state.novel, ...payload }),
      });
      await readStream(res, {
        onToken: (t) => {
          draftEditor.innerText += t;
          draftEditor.scrollTop = draftEditor.scrollHeight;
          updateDraftCount();
        },
      });
    } else {
      busy(true, "Revising…");
      const data = await apiPost("/api/revise", payload);
      draftEditor.innerText = data.content;
      updateDraftCount();
    }
    $("feedbackText").value = "";
    toast("Revised.", "success");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    busy(false);
    buttons.forEach((id) => { $(id).disabled = false; });
  }
}

$("btnRevise").addEventListener("click", () => doRevise(false));
$("btnReviseStream").addEventListener("click", () => doRevise(true));

$("btnCopyDraft").addEventListener("click", async () => {
  const text = draftEditor.innerText;
  if (!text) { toast("Nothing to copy yet.", "error"); return; }
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied.", "success");
  } catch {
    toast("Your browser blocked the clipboard.", "error");
  }
});

$("btnFinalize").addEventListener("click", async () => {
  const content = draftEditor.innerText.trim();
  if (!content) { toast("There's no draft to save.", "error"); return; }
  const number = parseInt($("chapterNumber").value, 10) || 1;

  busy(true, "Saving…");
  try {
    const data = await apiPost("/api/chapters/save", {
      chapter_number: number,
      title: $("chapterTitle").value.trim(),
      content,
    });
    toast(`Chapter ${number} saved and indexed (${data.chunks_ingested} segments).`, "success");
    $("chapterNumber").value = number + 1;
    $("chapterTitle").value = "";
    await refreshCounts();
  } catch (err) {
    toast(err.message, "error");
  } finally {
    busy(false);
  }
});

/* ============================================================
   Manuscript
   ============================================================ */

const writeEditor = $("writeEditor");

function updateWriteStats() {
  const text = writeEditor.innerText || "";
  $("writeWordCount").textContent = countWords(text);
  $("writeCharCount").textContent = text.length;
}
writeEditor.addEventListener("input", updateWriteStats);

async function autoDetectNextChapter() {
  try {
    const data = await apiGet("/api/chapters");
    const nums = data.chapters.map((ch) => {
      const m = ch.filename.match(/chapter-(\d+)/);
      return m ? parseInt(m[1], 10) : 0;
    });
    $("writeChapterNumber").value = nums.length ? Math.max(...nums) + 1 : 1;
  } catch { /* a fresh novel simply has none */ }
}

$("btnSaveManualChapter").addEventListener("click", async () => {
  const content = writeEditor.innerText.trim();
  if (!content) { toast("Write something first.", "error"); return; }

  const number = parseInt($("writeChapterNumber").value, 10) || 1;
  const title = $("writeChapterTitle").value.trim();

  const ok = await dialog({
    title: "Save chapter",
    confirmLabel: "Save & index",
    body: `<p>The chapter is filed under this novel and added to its memory, so later chapters can retrieve it.</p>
           <div class="dialog-detail">
             <b>Chapter ${number}${title ? ": " + escapeHtml(title) : ""}</b>
             <span>${countWords(content)} words · ${escapeHtml(currentNovel()?.title || "")}</span>
           </div>`,
  });
  if (!ok) return;

  busy(true, "Saving…");
  try {
    const data = await apiPost("/api/chapters/save", { chapter_number: number, title, content });
    toast(`Chapter ${number} saved (${data.chunks_ingested} segments indexed).`, "success");
    $("writeChapterNumber").value = number + 1;
    $("writeChapterTitle").value = "";
    writeEditor.innerText = "";
    updateWriteStats();
    await refreshCounts();
  } catch (err) {
    toast(err.message, "error");
  } finally {
    busy(false);
  }
});

/* ============================================================
   Chapters
   ============================================================ */

async function loadChapters() {
  const container = $("chaptersList");
  $("chapterReader").hidden = true;
  container.hidden = false;
  try {
    const data = await apiGet("/api/chapters");
    if (!data.chapters.length) {
      container.innerHTML = emptyState(ICON_BOOK, "No chapters yet",
        "Chapters you write or generate for this novel appear here.");
      return;
    }
    container.innerHTML = data.chapters.map((ch) => {
      const m = ch.filename.match(/chapter-(\d+)/);
      return `<div class="row" data-file="${escapeHtml(ch.filename)}">
        <span class="row-num">${m ? parseInt(m[1], 10) : "–"}</span>
        <div class="row-body">
          <div class="row-title">${escapeHtml(ch.title)}</div>
          <div class="row-meta">${ch.words.toLocaleString()} words · ${(ch.size / 1024).toFixed(1)} KB</div>
        </div>
        <button class="row-del" data-del="${escapeHtml(ch.filename)}" title="Delete chapter" type="button">${ICON_TRASH}</button>
      </div>`;
    }).join("");

    container.querySelectorAll(".row").forEach((row) => {
      row.addEventListener("click", (e) => {
        if (e.target.closest("[data-del]")) return;
        openChapter(row.dataset.file);
      });
    });
    container.querySelectorAll("[data-del]").forEach((btn) => {
      btn.addEventListener("click", (e) => { e.stopPropagation(); deleteChapter(btn.dataset.del); });
    });
  } catch (err) {
    toast(err.message, "error");
  }
}

async function openChapter(filename) {
  try {
    const data = await apiGet(`/api/chapters/${encodeURIComponent(filename)}`);
    $("chaptersList").hidden = true;
    $("chapterReader").hidden = false;
    $("chapterContent").textContent = data.content;
  } catch (err) {
    toast(err.message, "error");
  }
}

async function deleteChapter(filename) {
  const ok = await dialog({
    title: "Delete chapter?",
    confirmLabel: "Delete",
    danger: true,
    body: `<p>Removes the file and drops it from this novel's memory, so it stops appearing in retrieval.</p>
           <div class="dialog-detail"><b>${escapeHtml(filename)}</b></div>`,
  });
  if (!ok) return;
  try {
    await apiDelete(`/api/chapters/${encodeURIComponent(filename)}`);
    toast("Chapter deleted.", "success");
    await loadChapters();
    await refreshCounts();
  } catch (err) {
    toast(err.message, "error");
  }
}

$("btnBackToList").addEventListener("click", () => {
  $("chapterReader").hidden = true;
  $("chaptersList").hidden = false;
});

/* ============================================================
   World bible
   ============================================================ */

async function loadContextFiles() {
  const container = $("contextList");
  try {
    const data = await apiGet("/api/context");
    if (!data.files.length) {
      container.innerHTML = emptyState(ICON_GLOBE, "No world bible yet",
        "Add a file describing your characters, places and rules. Headings name entities.");
      return;
    }
    container.innerHTML = data.files.map((f) => `
      <div class="row${f.filename === state.contextFile ? " is-current" : ""}" data-file="${escapeHtml(f.filename)}">
        <div class="row-body">
          <div class="row-title">${escapeHtml(f.filename)}</div>
          <div class="row-meta">${(f.size / 1024).toFixed(1)} KB</div>
        </div>
      </div>`).join("");
    container.querySelectorAll(".row").forEach((row) => {
      row.addEventListener("click", () => openContextFile(row.dataset.file));
    });
  } catch (err) {
    toast(err.message, "error");
  }
}

async function openContextFile(filename) {
  try {
    const data = await apiGet(`/api/context/${encodeURIComponent(filename)}`);
    state.contextFile = data.filename;
    $("contextFilename").value = data.filename;
    $("contextEditor").value = data.content;
    $("btnDeleteContext").hidden = false;
    loadContextFiles();
  } catch (err) {
    toast(err.message, "error");
  }
}

$("btnNewContext").addEventListener("click", () => {
  state.contextFile = null;
  $("contextFilename").value = "";
  $("contextEditor").value = "";
  $("btnDeleteContext").hidden = true;
  $("contextFilename").focus();
  loadContextFiles();
});

$("btnSaveContext").addEventListener("click", async () => {
  let filename = $("contextFilename").value.trim();
  const content = $("contextEditor").value;
  if (!filename) { toast("Give the file a name, e.g. characters.md", "error"); return; }
  if (!content.trim()) { toast("The file is empty.", "error"); return; }
  if (!filename.toLowerCase().endsWith(".md")) filename += ".md";

  busy(true, "Indexing…");
  try {
    const data = await apiPut(`/api/context/${encodeURIComponent(filename)}`, { content });
    state.contextFile = data.filename;
    $("contextFilename").value = data.filename;
    $("btnDeleteContext").hidden = false;
    toast(`Saved and indexed ${data.segments} segments.`, "success");
    await loadContextFiles();
    await refreshCounts();
  } catch (err) {
    toast(err.message, "error");
  } finally {
    busy(false);
  }
});

$("btnDeleteContext").addEventListener("click", async () => {
  if (!state.contextFile) return;
  const ok = await dialog({
    title: "Delete this file?",
    confirmLabel: "Delete",
    danger: true,
    body: `<p>Removes the file and everything it contributed to this novel's memory.</p>
           <div class="dialog-detail"><b>${escapeHtml(state.contextFile)}</b></div>`,
  });
  if (!ok) return;
  try {
    await apiDelete(`/api/context/${encodeURIComponent(state.contextFile)}`);
    state.contextFile = null;
    $("contextFilename").value = "";
    $("contextEditor").value = "";
    $("btnDeleteContext").hidden = true;
    toast("File deleted.", "success");
    await loadContextFiles();
    await refreshCounts();
  } catch (err) {
    toast(err.message, "error");
  }
});

$("btnReingest").addEventListener("click", async () => {
  busy(true, "Re-indexing the world bible…");
  try {
    const data = await apiPost("/api/ingest/context", {});
    toast(`Re-indexed ${data.chunks_stored} segments.`, "success");
    await refreshCounts();
  } catch (err) {
    toast(err.message, "error");
  } finally {
    busy(false);
  }
});

/* ============================================================
   Memory
   ============================================================ */

async function loadMemory() {
  try {
    const s = await apiGet("/api/vectordb/stats");
    $("memMetrics").innerHTML = [
      ["Segments", s.segments],
      ["Sources", s.sources],
      ["Entities", s.entities],
      ["Relations", s.relations],
      ["Embedder", s.vector_space, true],
      ["Extractor", s.extractor, true],
    ].map(([k, v, small]) =>
      `<div class="metric"><span class="metric-k">${k}</span><span class="metric-v${small ? " sm" : ""}">${escapeHtml(v ?? "—")}</span></div>`
    ).join("");
  } catch (err) {
    toast(err.message, "error");
  }
}

async function refreshMemStat() {
  try {
    const s = await apiGet("/api/vectordb/stats");
    $("memStatText").textContent = `${s.segments} segment${s.segments === 1 ? "" : "s"}`;
    $("memDot").className = "dot " + (s.segments > 0 ? "ok" : "empty");
  } catch {
    $("memStatText").textContent = "unavailable";
    $("memDot").className = "dot";
  }
}

$("btnSearch").addEventListener("click", runSearch);
$("searchQuery").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });

async function runSearch() {
  const query = $("searchQuery").value.trim();
  if (!query) { toast("Type something to search for.", "error"); return; }
  const container = $("searchResults");
  try {
    const data = await apiPost("/api/search", { query, top_k: 8 });
    if (!data.results.length) {
      container.innerHTML = emptyState(ICON_SEARCH, "Nothing found",
        "This novel's memory has no passage matching that. Try indexing its world bible first.");
      return;
    }
    container.innerHTML = data.results.map((hit) => `
      <div class="hit">
        <div class="hit-text">${escapeHtml(hit.text)}</div>
        <div class="hit-meta">
          <span>${escapeHtml(hit.metadata.source)}</span>
          ${hit.metadata.heading ? `<span>${escapeHtml(hit.metadata.heading)}</span>` : ""}
          <span>${escapeHtml(hit.metadata.role)}</span>
          ${hit.metadata.matched_entities ? `<b>${escapeHtml(hit.metadata.matched_entities)}</b>` : ""}
        </div>
      </div>`).join("");
  } catch (err) {
    toast(err.message, "error");
  }
}

$("btnIngestText").addEventListener("click", async () => {
  const text = $("ingestText").value.trim();
  if (!text) { toast("Paste some text first.", "error"); return; }
  try {
    const data = await apiPost("/api/ingest/text", {
      text, source: $("ingestSource").value.trim() || "manual",
    });
    toast(`Added ${data.chunks_stored} segments.`, "success");
    $("ingestText").value = "";
    $("ingestSource").value = "";
    await loadMemory();
    await refreshMemStat();
  } catch (err) {
    toast(err.message, "error");
  }
});

$("btnClearVdb").addEventListener("click", async () => {
  const ok = await dialog({
    title: "Clear this novel's memory?",
    confirmLabel: "Clear memory",
    danger: true,
    body: `<p>Erases everything indexed for <b>${escapeHtml(currentNovel()?.title || "this novel")}</b>. Your other novels are untouched.</p>
           <p>Context and chapter files stay on disk, so <b>Re-index everything</b> rebuilds from them.</p>`,
  });
  if (!ok) return;
  try {
    await apiDelete("/api/vectordb/clear");
    toast("Memory cleared.", "success");
    await loadMemory();
    await refreshMemStat();
  } catch (err) {
    toast(err.message, "error");
  }
});

/* ============================================================
   Settings
   ============================================================ */

async function loadSettings() {
  try {
    const [config, rules, skills] = await Promise.all([
      apiGet("/api/config"), apiGet("/api/rules"), apiGet("/api/skills"),
    ]);

    $("configDisplay").innerHTML = Object.entries(config).map(([k, v]) => {
      const label = k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
      const cls = typeof v === "boolean" ? (v ? "yes" : "no") : "";
      const shown = typeof v === "boolean" ? (v ? "configured" : "not set") : (v === "" ? "—" : v);
      return `<dt>${escapeHtml(label)}</dt><dd class="${cls}">${escapeHtml(shown)}</dd>`;
    }).join("");

    const docs = (items) => items.length
      ? items.map((d) => `<details class="doc"><summary>${escapeHtml(d.name)}<span class="tag ${d.scope}">${d.scope}</span></summary><pre>${escapeHtml(d.content)}</pre></details>`).join("")
      : `<p class="card-note">None defined.</p>`;

    $("rulesDisplay").innerHTML = docs(rules.rules);
    $("skillsDisplay").innerHTML = docs(skills.skills);
  } catch (err) {
    toast(err.message, "error");
  }
}

/* ============================================================
   Chat
   ============================================================ */

const chatLog = $("chatMessages");
const chatInput = $("chatInput");

function chatWelcome() {
  const name = currentNovel()?.title || "this novel";
  return emptyState(
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
    `Ask about ${name}`,
    "Characters, plot threads, places, what happened when. Answers come only from this novel's memory — never from your other books."
  );
}

function resetChat() {
  state.chatHistory = [];
  chatLog.innerHTML = chatWelcome();
}

$("btnClearChat").addEventListener("click", resetChat);

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 128) + "px";
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

$("btnChatSend").addEventListener("click", sendChat);

$("chatShowSources").addEventListener("change", (e) => {
  document.querySelectorAll(".sources").forEach((el) => el.classList.toggle("visible", e.target.checked));
});

function addMessage(role, text) {
  const empty = chatLog.querySelector(".empty");
  if (empty) empty.remove();
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msg.appendChild(bubble);
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
  return { msg, bubble };
}

function addTyping() {
  const empty = chatLog.querySelector(".empty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `<div class="typing"><i></i><i></i><i></i></div>`;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return el;
}

function addSources(msgEl, sources) {
  if (!sources || !sources.length) return;
  const wrap = document.createElement("div");
  wrap.className = "sources" + ($("chatShowSources").checked ? " visible" : "");
  wrap.innerHTML = sources.map((s, i) =>
    `<div class="source"><b>[${i + 1}] ${escapeHtml(s.source)}</b> — ${escapeHtml(s.text.slice(0, 160))}…</div>`
  ).join("");
  msgEl.appendChild(wrap);
}

async function sendChat() {
  const message = chatInput.value.trim();
  if (!message) return;

  chatInput.value = "";
  chatInput.style.height = "auto";
  addMessage("user", message);
  const history = state.chatHistory.slice();
  state.chatHistory.push({ role: "user", content: message });

  const typing = addTyping();
  $("btnChatSend").disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        novel: state.novel, message, history, top_k: 5,
        provider: $("provider").value || null,
        model: $("model").value || null,
        stream: true,
      }),
    });

    typing.remove();
    const { msg, bubble } = addMessage("bot", "");
    let answer = "";
    let sources = [];

    await readStream(res, {
      onSources: (s) => { sources = s; },
      onToken: (t) => {
        answer += t;
        bubble.textContent = answer;
        chatLog.scrollTop = chatLog.scrollHeight;
      },
    });

    state.chatHistory.push({ role: "assistant", content: answer });
    addSources(msg, sources);
  } catch (err) {
    typing.remove();
    toast(err.message, "error");
  } finally {
    $("btnChatSend").disabled = false;
    chatInput.focus();
  }
}

/* ============================================================
   Refresh + init
   ============================================================ */

async function refreshCounts() {
  try {
    const data = await apiGet("/api/novels");
    state.novels = data.novels;
    reflectNovel();
    await refreshMemStat();
  } catch { /* the sidebar counts are cosmetic */ }
}

async function refreshAll() {
  const active = document.querySelector(".panel.is-active");
  await refreshMemStat();
  if (!active) return;
  const name = active.id.replace("panel-", "");
  if (name === "chapters") await loadChapters();
  if (name === "world") await loadContextFiles();
  if (name === "memory") await loadMemory();
  if (name === "settings") await loadSettings();
  if (name === "manuscript") await autoDetectNextChapter();
}

(async function init() {
  resetChat();
  try {
    await loadNovels();
    resetChat();               // the welcome line names the novel
    await refreshMemStat();
  } catch (err) {
    toast(`Could not reach the server: ${err.message}`, "error");
  }
})();
