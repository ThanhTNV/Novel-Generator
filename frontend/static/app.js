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

        if (panelId === "write") autoDetectNextChapter();
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
        // max_tokens omitted on purpose: the server sizes the output budget
        // from target_words. Vietnamese runs ~2.4 tokens/word, so the old
        // fixed 4096 cut every default-length chapter off mid-sentence.
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
        // omitted: the server sizes the budget from the draft's own length,
        // since a revision returns the complete rewritten chapter
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

// ---- Chat ----

const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
let chatHistory = [];

chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});

chatInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

document.getElementById("btnChatSend").addEventListener("click", sendChatMessage);

document.getElementById("btnClearChat").addEventListener("click", () => {
    chatHistory = [];
    chatMessages.innerHTML = `
        <div class="chat-welcome">
            <div class="chat-welcome-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            </div>
            <p>Ask anything about your story — characters, plot, locations, events. Answers are grounded in your vector database.</p>
        </div>`;
});

function appendChatMsg(role, text) {
    const welcome = chatMessages.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const msg = document.createElement("div");
    msg.className = `chat-msg ${role}`;
    msg.innerHTML = `<div class="chat-msg-bubble">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
}

function appendTypingIndicator() {
    const welcome = chatMessages.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const msg = document.createElement("div");
    msg.className = "chat-msg assistant";
    msg.id = "chatTyping";
    msg.innerHTML = `<div class="chat-msg-typing"><span></span><span></span><span></span></div>`;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function addSourcesBlock(msgEl, sources) {
    if (!sources || !sources.length) return;
    const showSources = document.getElementById("chatShowSources").checked;
    const srcDiv = document.createElement("div");
    srcDiv.className = `chat-msg-sources${showSources ? " visible" : ""}`;
    srcDiv.innerHTML = sources.map((s, i) =>
        `<strong>[${i + 1}] ${s.source}</strong>: ${escapeHtml(s.text.substring(0, 120))}...`
    ).join("<br>");
    msgEl.appendChild(srcDiv);
}

document.getElementById("chatShowSources").addEventListener("change", e => {
    const visible = e.target.checked;
    document.querySelectorAll(".chat-msg-sources").forEach(el => {
        el.classList.toggle("visible", visible);
    });
});

async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    chatInput.value = "";
    chatInput.style.height = "auto";

    appendChatMsg("user", message);
    chatHistory.push({ role: "user", content: message });

    const typingEl = appendTypingIndicator();

    const payload = {
        message,
        history: chatHistory.slice(0, -1),
        top_k: 5,
        provider: document.getElementById("provider").value || null,
        model: document.getElementById("model").value || null,
        stream: true,
    };

    document.getElementById("btnChatSend").disabled = true;

    try {
        const res = await fetch(`${API}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        typingEl.remove();

        const assistantMsg = document.createElement("div");
        assistantMsg.className = "chat-msg assistant";
        const bubble = document.createElement("div");
        bubble.className = "chat-msg-bubble";
        assistantMsg.appendChild(bubble);
        chatMessages.appendChild(assistantMsg);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let fullAnswer = "";
        let sources = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const data = line.slice(6);
                if (data === "[DONE]") continue;
                try {
                    const parsed = JSON.parse(data);
                    if (parsed.sources) {
                        sources = parsed.sources;
                    }
                    if (parsed.token) {
                        fullAnswer += parsed.token;
                        bubble.textContent = fullAnswer;
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                } catch {}
            }
        }

        chatHistory.push({ role: "assistant", content: fullAnswer });
        addSourcesBlock(assistantMsg, sources);

    } catch (err) {
        typingEl.remove();
        toast(`Chat error: ${err.message}`, "error");
    } finally {
        document.getElementById("btnChatSend").disabled = false;
        chatInput.focus();
    }
}

// ---- Confirmation Modal ----

function showConfirmModal(title, bodyHTML) {
    return new Promise((resolve) => {
        document.getElementById("confirmModalTitle").textContent = title;
        document.getElementById("confirmModalBody").innerHTML = bodyHTML;
        const modal = document.getElementById("confirmModal");
        modal.style.display = "flex";

        function cleanup(result) {
            modal.style.display = "none";
            document.getElementById("confirmModalOk").removeEventListener("click", onOk);
            document.getElementById("confirmModalCancel").removeEventListener("click", onCancel);
            modal.removeEventListener("click", onBackdrop);
            resolve(result);
        }
        function onOk() { cleanup(true); }
        function onCancel() { cleanup(false); }
        function onBackdrop(e) { if (e.target === modal) cleanup(false); }

        document.getElementById("confirmModalOk").addEventListener("click", onOk);
        document.getElementById("confirmModalCancel").addEventListener("click", onCancel);
        modal.addEventListener("click", onBackdrop);
    });
}

// ---- Write Panel ----

const writeEditor = document.getElementById("writeEditor");

function updateWriteStats() {
    const text = writeEditor.innerText || "";
    const words = text.trim().split(/\s+/).filter(Boolean).length;
    const chars = text.length;
    document.getElementById("writeWordCount").textContent = words;
    document.getElementById("writeCharCount").textContent = chars;
}

writeEditor.addEventListener("input", updateWriteStats);

async function autoDetectNextChapter() {
    try {
        const data = await apiGet("/api/chapters");
        const nums = data.chapters.map(ch => {
            const m = ch.filename.match(/chapter-(\d+)/);
            return m ? parseInt(m[1]) : 0;
        });
        const next = nums.length ? Math.max(...nums) + 1 : 1;
        document.getElementById("writeChapterNumber").value = next;
    } catch {}
}

document.getElementById("btnSaveManualChapter").addEventListener("click", async () => {
    const content = writeEditor.innerText.trim();
    if (!content) {
        toast("Please write some content before saving.", "error");
        return;
    }

    const chapterNum = parseInt(document.getElementById("writeChapterNumber").value) || 1;
    const chapterTitle = document.getElementById("writeChapterTitle").value.trim();
    const wordCount = content.split(/\s+/).filter(Boolean).length;

    const titleDisplay = chapterTitle || "(Untitled)";
    const confirmed = await showConfirmModal(
        "Save Chapter",
        `<p>You are about to save this chapter. It will be stored as a file and ingested into the vector database for RAG retrieval.</p>
         <div class="modal-detail">
             <strong>Chapter ${chapterNum}:</strong> ${titleDisplay}<br>
             <strong>Word count:</strong> ${wordCount} words
         </div>`
    );

    if (!confirmed) return;

    showLoading("Saving chapter...");
    try {
        const data = await apiPost("/api/chapters/save", {
            chapter_number: chapterNum,
            title: chapterTitle,
            content,
        });
        toast(`Chapter ${chapterNum} saved! ${data.chunks_ingested} chunks added to vector store.`, "success");
        document.getElementById("writeChapterNumber").value = chapterNum + 1;
        document.getElementById("writeChapterTitle").value = "";
        writeEditor.innerText = "";
        updateWriteStats();
    } catch (err) {
        toast(`Error: ${err.message}`, "error");
    } finally {
        hideLoading();
    }
});

// ---- Init ----

(async function init() {
    loadVdbStats();
})();
