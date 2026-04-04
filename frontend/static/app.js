/* ============================================================
   Novel Generator – Frontend Application
   ============================================================ */

const API = "";

// ---- Utility ----

function toast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

function showLoading(text = "Generating...") {
    document.getElementById("loadingText").textContent = text;
    document.getElementById("loadingOverlay").style.display = "flex";
}

function hideLoading() {
    document.getElementById("loadingOverlay").style.display = "none";
}

async function apiPost(endpoint, body) {
    const res = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
    }
    return res.json();
}

async function apiGet(endpoint) {
    const res = await fetch(`${API}${endpoint}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

async function apiDelete(endpoint) {
    const res = await fetch(`${API}${endpoint}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

function parseList(str) {
    return str.split(",").map(s => s.trim()).filter(Boolean);
}

function countWords(text) {
    return text.trim().split(/\s+/).filter(Boolean).length;
}

function updateWordCount() {
    const editor = document.getElementById("draftEditor");
    const text = editor.innerText || "";
    const count = countWords(text);
    document.getElementById("wordCount").textContent = `${count} word${count !== 1 ? "s" : ""}`;
}

// ---- Navigation ----

document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const panelId = btn.dataset.panel;
        document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
        document.getElementById(`panel-${panelId}`).classList.add("active");

        if (panelId === "chapters") loadChapters();
        if (panelId === "vectordb") loadVdbStats();
        if (panelId === "settings") loadSettings();
    });
});

// ---- Temperature slider ----

document.getElementById("temperature").addEventListener("input", e => {
    document.getElementById("tempValue").textContent = e.target.value;
});

// ---- Draft editor word count ----

document.getElementById("draftEditor").addEventListener("input", updateWordCount);

// ---- Generate ----

function getGeneratePayload(stream = false) {
    return {
        chapter_instructions: document.getElementById("chapterInstructions").value,
        story_summary: document.getElementById("storySummary").value,
        characters: parseList(document.getElementById("characters").value),
        locations: parseList(document.getElementById("locations").value),
        target_words: parseInt(document.getElementById("targetWords").value) || 2000,
        temperature: parseFloat(document.getElementById("temperature").value),
        max_tokens: 4096,
        provider: document.getElementById("provider").value || null,
        model: document.getElementById("model").value || null,
        stream,
    };
}

document.getElementById("btnGenerate").addEventListener("click", async () => {
    const payload = getGeneratePayload(false);
    if (!payload.chapter_instructions) {
        toast("Please provide chapter instructions.", "error");
        return;
    }
    showLoading("Generating chapter draft...");
    try {
        const data = await apiPost("/api/generate", payload);
        document.getElementById("draftEditor").innerText = data.content;
        updateWordCount();
        document.getElementById("feedbackSection").style.display = "block";
        document.getElementById("finalizeSection").style.display = "block";
        toast("Draft generated!", "success");
    } catch (err) {
        toast(`Error: ${err.message}`, "error");
    } finally {
        hideLoading();
    }
});

document.getElementById("btnStream").addEventListener("click", async () => {
    const payload = getGeneratePayload(true);
    if (!payload.chapter_instructions) {
        toast("Please provide chapter instructions.", "error");
        return;
    }

    const editor = document.getElementById("draftEditor");
    editor.innerText = "";

    document.getElementById("btnStream").disabled = true;
    document.getElementById("btnGenerate").disabled = true;

    try {
        const res = await fetch(`${API}/api/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const data = line.slice(6);
                    if (data === "[DONE]") continue;
                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.token) {
                            editor.innerText += parsed.token;
                            editor.scrollTop = editor.scrollHeight;
                        }
                    } catch {}
                }
            }
        }

        updateWordCount();
        document.getElementById("feedbackSection").style.display = "block";
        document.getElementById("finalizeSection").style.display = "block";
        toast("Draft streamed!", "success");
    } catch (err) {
        toast(`Stream error: ${err.message}`, "error");
    } finally {
        document.getElementById("btnStream").disabled = false;
        document.getElementById("btnGenerate").disabled = false;
    }
});

// ---- Revise ----

async function doRevise(stream) {
    const draft = document.getElementById("draftEditor").innerText;
    const feedback = document.getElementById("feedbackText").value;
    if (!draft) { toast("No draft to revise.", "error"); return; }
    if (!feedback) { toast("Please provide feedback.", "error"); return; }

    const payload = {
        draft,
        feedback,
        temperature: 0.5,
        max_tokens: 4096,
        provider: document.getElementById("provider").value || null,
        model: document.getElementById("model").value || null,
        stream,
    };

    if (!stream) {
        showLoading("Revising draft...");
        try {
            const data = await apiPost("/api/revise", payload);
            document.getElementById("draftEditor").innerText = data.content;
            updateWordCount();
            document.getElementById("feedbackText").value = "";
            toast("Draft revised!", "success");
        } catch (err) {
            toast(`Error: ${err.message}`, "error");
        } finally {
            hideLoading();
        }
    } else {
        const editor = document.getElementById("draftEditor");
        editor.innerText = "";
        document.getElementById("btnRevise").disabled = true;
        document.getElementById("btnReviseStream").disabled = true;

        try {
            const res = await fetch(`${API}/api/revise`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6);
                        if (data === "[DONE]") continue;
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.token) {
                                editor.innerText += parsed.token;
                                editor.scrollTop = editor.scrollHeight;
                            }
                        } catch {}
                    }
                }
            }
            updateWordCount();
            document.getElementById("feedbackText").value = "";
            toast("Revision streamed!", "success");
        } catch (err) {
            toast(`Stream error: ${err.message}`, "error");
        } finally {
            document.getElementById("btnRevise").disabled = false;
            document.getElementById("btnReviseStream").disabled = false;
        }
    }
}

document.getElementById("btnRevise").addEventListener("click", () => doRevise(false));
document.getElementById("btnReviseStream").addEventListener("click", () => doRevise(true));

// ---- Copy draft ----

document.getElementById("btnCopyDraft").addEventListener("click", () => {
    const text = document.getElementById("draftEditor").innerText;
    if (!text) { toast("Nothing to copy.", "error"); return; }
    navigator.clipboard.writeText(text).then(
        () => toast("Copied to clipboard!", "success"),
        () => toast("Copy failed.", "error")
    );
});

// ---- Finalize ----

document.getElementById("btnFinalize").addEventListener("click", async () => {
    const content = document.getElementById("draftEditor").innerText;
    if (!content) { toast("No draft to save.", "error"); return; }

    const payload = {
        chapter_number: parseInt(document.getElementById("chapterNumber").value) || 1,
        title: document.getElementById("chapterTitle").value,
        content,
    };

    showLoading("Saving chapter...");
    try {
        const data = await apiPost("/api/chapters/save", payload);
        toast(`Chapter saved! ${data.chunks_ingested} chunks added to vector store.`, "success");
        document.getElementById("chapterNumber").value = payload.chapter_number + 1;
    } catch (err) {
        toast(`Error: ${err.message}`, "error");
    } finally {
        hideLoading();
    }
});

// ---- Chapters ----

async function loadChapters() {
    try {
        const data = await apiGet("/api/chapters");
        const container = document.getElementById("chaptersList");

        if (!data.chapters.length) {
            container.innerHTML = '<p class="empty-state">No chapters saved yet.</p>';
            return;
        }

        container.innerHTML = data.chapters.map(ch => `
            <div class="chapter-item" data-filename="${ch.filename}">
                <span class="chapter-item-title">${ch.title}</span>
                <span class="chapter-item-meta">${(ch.size / 1024).toFixed(1)} KB</span>
            </div>
        `).join("");

        container.querySelectorAll(".chapter-item").forEach(item => {
            item.addEventListener("click", () => openChapter(item.dataset.filename));
        });
    } catch (err) {
        toast(`Error loading chapters: ${err.message}`, "error");
    }
}

async function openChapter(filename) {
    try {
        const data = await apiGet(`/api/chapters/${filename}`);
        document.getElementById("chaptersList").style.display = "none";
        document.getElementById("chapterReader").style.display = "block";
        document.getElementById("chapterContent").innerText = data.content;
    } catch (err) {
        toast(`Error: ${err.message}`, "error");
    }
}

document.getElementById("btnBackToList").addEventListener("click", () => {
    document.getElementById("chapterReader").style.display = "none";
    document.getElementById("chaptersList").style.display = "flex";
});

document.getElementById("btnRefreshChapters").addEventListener("click", loadChapters);

// ---- Vector DB ----

async function loadVdbStats() {
    try {
        const data = await apiGet("/api/vectordb/stats");
        document.getElementById("vdbStats").textContent = `${data.count} chunks`;
        document.getElementById("dbStatus").querySelector(".status-text").textContent = `${data.count} chunks`;
    } catch (err) {
        document.getElementById("vdbStats").textContent = "Error";
    }
}

document.getElementById("btnIngestContext").addEventListener("click", async () => {
    showLoading("Ingesting context files...");
    try {
        const data = await apiPost("/api/ingest/context", {});
        toast(`Ingested ${data.chunks_stored} chunks from context directory.`, "success");
        loadVdbStats();
    } catch (err) {
        toast(`Error: ${err.message}`, "error");
    } finally {
        hideLoading();
    }
});

document.getElementById("btnIngestText").addEventListener("click", async () => {
    const text = document.getElementById("ingestText").value;
    const source = document.getElementById("ingestSource").value || "manual";
    if (!text) { toast("Please enter text to ingest.", "error"); return; }

    try {
        const data = await apiPost("/api/ingest/text", { text, source });
        toast(`Ingested ${data.chunks_stored} chunks.`, "success");
        document.getElementById("ingestText").value = "";
        loadVdbStats();
    } catch (err) {
        toast(`Error: ${err.message}`, "error");
    }
});

document.getElementById("btnSearch").addEventListener("click", async () => {
    const query = document.getElementById("searchQuery").value;
    if (!query) { toast("Enter a search query.", "error"); return; }

    try {
        const data = await apiPost("/api/search", { query, top_k: 8 });
        const container = document.getElementById("searchResults");

        if (!data.results.length) {
            container.innerHTML = '<p class="empty-state">No results found.</p>';
            return;
        }

        container.innerHTML = data.results.map(hit => `
            <div class="search-hit">
                ${hit.text}
                <div class="search-hit-meta">
                    Source: ${hit.metadata?.source || "unknown"} | Distance: ${hit.distance?.toFixed(4) || "N/A"}
                </div>
            </div>
        `).join("");
    } catch (err) {
        toast(`Search error: ${err.message}`, "error");
    }
});

document.getElementById("btnClearVdb").addEventListener("click", async () => {
    if (!confirm("Clear all data from the vector store? This cannot be undone.")) return;
    try {
        await apiDelete("/api/vectordb/clear");
        toast("Vector store cleared.", "success");
        loadVdbStats();
    } catch (err) {
        toast(`Error: ${err.message}`, "error");
    }
});

// ---- Settings ----

async function loadSettings() {
    try {
        const [config, rules, skills] = await Promise.all([
            apiGet("/api/config"),
            apiGet("/api/rules"),
            apiGet("/api/skills"),
        ]);

        document.getElementById("configDisplay").textContent = JSON.stringify(config, null, 2);

        const rulesDiv = document.getElementById("rulesDisplay");
        if (rules.rules.length) {
            rulesDiv.innerHTML = rules.rules.map(r => `
                <div class="rule-item">
                    <h4>${r.name}</h4>
                    <pre>${r.content}</pre>
                </div>
            `).join("");
        } else {
            rulesDiv.innerHTML = '<p class="empty-state">No rules defined.</p>';
        }

        const skillsDiv = document.getElementById("skillsDisplay");
        if (skills.skills.length) {
            skillsDiv.innerHTML = skills.skills.map(s => `
                <div class="skill-item">
                    <h4>${s.name}</h4>
                    <pre>${s.content}</pre>
                </div>
            `).join("");
        } else {
            skillsDiv.innerHTML = '<p class="empty-state">No skills defined.</p>';
        }
    } catch (err) {
        toast(`Error loading settings: ${err.message}`, "error");
    }
}

// ---- Init ----

(async function init() {
    loadVdbStats();
})();
