/* ====================================================================
   KATOMARAN HACKATHON: INTELLIGENT FACE TRACKER & VISITOR COUNTER
   High-Frequency Real-Time Controller with Decoupled Telemetry Polling
   ==================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

let state = {
    activeTab: "visitors",
    chartData: { labels: [], cpu: [], fps: [], active: [] },
    config: {},
    availableVideos: [],
    lastVisitorsHash: "",
    lastEventsHash: "",
    lastLogsHash: ""
};

function initApp() {
    setupTabNavigation();
    setupModals();
    setupActionButtons();
    setupVideoFeed();
    fetchVideosList();
    fetchConfig();
    initChart();
    
    // High-frequency 200ms polling cycle for live visitor counts & compute stats
    pollLiveState();
    setInterval(pollLiveState, 200);
}

/* Tab Navigation */
function setupTabNavigation() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const target = btn.dataset.tab;
            state.activeTab = target;
            
            document.querySelectorAll(".tab-pane").forEach(pane => {
                pane.style.display = pane.id === `tab-${target}` ? "block" : "none";
            });
            
            // Force re-render for newly visible tab
            state.lastVisitorsHash = "";
            state.lastEventsHash = "";
            state.lastLogsHash = "";
            pollLiveState();
        });
    });
}

/* Action Buttons (Reset Video & Clear DB) */
function setupActionButtons() {
    const resetBtn = document.getElementById("btn-reset-session");
    if (resetBtn) {
        resetBtn.onclick = async () => {
            if (confirm("Reset current video tracking counts and clear active tracks?")) {
                // Immediate UI reset
                updateCountersDirect(0, 0, 0, 0);
                await fetch("/api/session/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ clear_db: false })
                });
                refreshVideoStream();
                pollLiveState();
            }
        };
    }

    const clearDbBtn = document.getElementById("btn-clear-db");
    if (clearDbBtn) {
        clearDbBtn.onclick = async () => {
            if (confirm("⚠️ Clear ALL database records, registered visitors, and event logs?")) {
                updateCountersDirect(0, 0, 0, 0);
                await fetch("/api/session/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ clear_db: true })
                });
                state.lastVisitorsHash = "";
                state.lastEventsHash = "";
                state.lastLogsHash = "";
                refreshVideoStream();
                pollLiveState();
            }
        };
    }
}

function updateCountersDirect(unique, active, entries, exits) {
    const kpiUnique = document.getElementById("kpi-unique-visitors");
    const kpiActive = document.getElementById("kpi-active-frame");
    const kpiEntries = document.getElementById("kpi-today-entries");
    const kpiExits = document.getElementById("kpi-today-exits");
    if (kpiUnique) kpiUnique.textContent = unique;
    if (kpiActive) kpiActive.textContent = active;
    if (kpiEntries) kpiEntries.textContent = entries;
    if (kpiExits) kpiExits.textContent = exits;
}

/* Modals & Lightbox */
function setupModals() {
    // Config Modal
    const configBtn = document.getElementById("btn-open-config");
    const configModal = document.getElementById("modal-config");
    const configClose = document.getElementById("close-config");
    const configForm = document.getElementById("form-config");
    
    if (configBtn) configBtn.onclick = () => { configModal.classList.add("active"); };
    if (configClose) configClose.onclick = () => { configModal.classList.remove("active"); };
    
    if (configForm) {
        configForm.onsubmit = async (e) => {
            e.preventDefault();
            const frameSkip = parseInt(document.getElementById("cfg-frame-skip").value);
            const simThresh = parseFloat(document.getElementById("cfg-sim-thresh").value);
            const confThresh = parseFloat(document.getElementById("cfg-conf-thresh").value);
            
            await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    detection: { frame_skip: frameSkip, confidence_threshold: confThresh },
                    recognition: { similarity_threshold: simThresh }
                })
            });
            configModal.classList.remove("active");
            fetchConfig();
        };
    }
    
    // RTSP Switch Modal
    const rtspBtn = document.getElementById("btn-open-rtsp");
    const rtspModal = document.getElementById("modal-rtsp");
    const rtspClose = document.getElementById("close-rtsp");
    const rtspForm = document.getElementById("form-rtsp");
    
    if (rtspBtn) rtspBtn.onclick = () => { rtspModal.classList.add("active"); };
    if (rtspClose) rtspClose.onclick = () => { rtspModal.classList.remove("active"); };
    
    if (rtspForm) {
        rtspForm.onsubmit = async (e) => {
            e.preventDefault();
            const rtspUrl = document.getElementById("rtsp-url-input").value;
            if (!rtspUrl) return;
            
            updateCountersDirect(0, 0, 0, 0);
            await fetch("/api/stream/switch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source: rtspUrl, is_rtsp: true })
            });
            rtspModal.classList.remove("active");
            state.lastVisitorsHash = "";
            state.lastEventsHash = "";
            refreshVideoStream();
            pollLiveState();
        };
    }

    // Lightbox Modal
    const lightbox = document.getElementById("lightbox-modal");
    const lightboxClose = document.getElementById("close-lightbox");
    if (lightboxClose) lightboxClose.onclick = () => { lightbox.classList.remove("active"); };
    if (lightbox) lightbox.onclick = (e) => { if (e.target === lightbox) lightbox.classList.remove("active"); };
}

function openLightbox(imgSrc, title) {
    const lightbox = document.getElementById("lightbox-modal");
    const img = document.getElementById("lightbox-img");
    const caption = document.getElementById("lightbox-caption");
    if (img && lightbox) {
        img.src = imgSrc;
        if (caption) caption.textContent = title;
        lightbox.classList.add("active");
    }
}

function setupVideoFeed() {
    const img = document.getElementById("main-video-feed");
    if (img) {
        img.onerror = () => {
            setTimeout(() => {
                img.src = `/video_feed?t=${Date.now()}`;
            }, 600);
        };
    }
}

/* Polling & Unified Live State with DOM Diffing */
let lastChartPush = 0;
let isPolling = false;

async function pollLiveState() {
    if (isPolling) return;
    isPolling = true;

    try {
        const res = await fetch("/api/live_state");
        if (!res.ok) return;
        const data = await res.json();
        
        const stats = data.stats || {};
        const comp = stats.compute || {};
        
        // 1. Update Top KPI Cards Instantly
        const kpiUnique = document.getElementById("kpi-unique-visitors");
        const kpiActive = document.getElementById("kpi-active-frame");
        const kpiEntries = document.getElementById("kpi-today-entries");
        const kpiExits = document.getElementById("kpi-today-exits");
        const kpiFps = document.getElementById("kpi-fps");
        
        if (kpiUnique) kpiUnique.textContent = stats.session_unique_visitors !== undefined ? stats.session_unique_visitors : (stats.unique_visitors || 0);
        if (kpiActive) kpiActive.textContent = stats.active_in_frame || 0;
        if (kpiEntries) kpiEntries.textContent = stats.session_entries !== undefined ? stats.session_entries : (stats.today_entries || 0);
        if (kpiExits) kpiExits.textContent = stats.session_exits !== undefined ? stats.session_exits : (stats.today_exits || 0);
        if (kpiFps) kpiFps.textContent = (stats.current_fps || 0).toFixed(1);
        
        // 2. Update Telemetry Gauges
        const cpuVal = document.getElementById("gauge-cpu-val");
        const cpuBar = document.getElementById("gauge-cpu-bar");
        const ramVal = document.getElementById("gauge-ram-val");
        const gpuVal = document.getElementById("gauge-gpu-val");
        
        if (cpuVal) cpuVal.textContent = `${comp.cpu_percent || 0}%`;
        if (cpuBar) cpuBar.style.width = `${Math.min(100, comp.cpu_percent || 0)}%`;
        if (ramVal) ramVal.textContent = `${comp.memory_mb || 0} MB`;
        if (gpuVal) gpuVal.textContent = comp.gpu_name || "CPU Optimized";
        
        const latTotal = document.getElementById("latency-total-val");
        const latDet = document.getElementById("latency-det-val");
        const latRec = document.getElementById("latency-rec-val");
        
        if (latTotal) latTotal.textContent = `${comp.frame_process_time_ms || 0} ms`;
        if (latDet) latDet.textContent = `${comp.detection_time_ms || 0} ms`;
        if (latRec) latRec.textContent = `${comp.recognition_time_ms || 0} ms`;
        
        // 3. Render Active Tab Content with Diff Hash Check
        if (state.activeTab === "visitors" && data.visitors) {
            const hash = data.visitors.map(v => `${v.visitor_id}_${v.total_visits}_${v.last_seen}`).join("|");
            if (hash !== state.lastVisitorsHash) {
                state.lastVisitorsHash = hash;
                renderVisitors(data.visitors);
            }
        } else if (state.activeTab === "events" && data.events) {
            const hash = data.events.map(e => `${e.event_id}_${e.event_type}`).join("|");
            if (hash !== state.lastEventsHash) {
                state.lastEventsHash = hash;
                renderEvents(data.events);
            }
        } else if (state.activeTab === "logs" && data.logs) {
            const hash = data.logs.slice(-10).join("|");
            if (hash !== state.lastLogsHash) {
                state.lastLogsHash = hash;
                renderTerminalLogs(data.logs);
            }
        }
        
        // 4. Smooth Chart Updates (every 1s)
        const now = Date.now();
        if (now - lastChartPush >= 1000) {
            lastChartPush = now;
            const timeLabel = new Date().toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' });
            pushChartData(timeLabel, comp.cpu_percent || 0, stats.current_fps || 0, stats.active_in_frame || 0);
        }
        
    } catch (e) {
        // Silently catch temporary network hiccups
    } finally {
        isPolling = false;
    }
}

function renderVisitors(visitors) {
    const container = document.getElementById("visitors-grid-container");
    if (!container) return;
    
    if (visitors.length === 0) {
        container.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-dim);">
                <p>No visitors registered in current session. Processing video...</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = visitors.map(v => {
        const firstSeen = new Date(v.first_seen).toLocaleTimeString();
        const lastSeen = new Date(v.last_seen).toLocaleTimeString();
        const thumb = v.thumbnail_path ? `/${v.thumbnail_path}` : 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" fill="%2364748b" viewBox="0 0 16 16"><path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10c-2.29 0-3.516.68-4.168 1.332-.678.678-.83 1.418-.832 1.664h10z"/></svg>';
        
        return `
            <div class="visitor-card">
                <div class="visitor-avatar-wrapper">
                    <img src="${thumb}" alt="${v.visitor_id}" class="visitor-avatar" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'96\\' height=\\'96\\' fill=\\'%2364748b\\' viewBox=\\'0 0 16 16\\'><path d=\\'M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10c-2.29 0-3.516.68-4.168 1.332-.678.678-.83 1.418-.832 1.664h10z\\'/></svg>'">
                </div>
                <div class="visitor-id">${v.visitor_id}</div>
                <div class="visitor-badge">${v.total_visits} ${v.total_visits === 1 ? 'Session' : 'Sessions'}</div>
                <div class="visitor-meta">
                    <div class="visitor-meta-row">
                        <span>First Seen:</span>
                        <strong style="color: #fff;">${firstSeen}</strong>
                    </div>
                    <div class="visitor-meta-row">
                        <span>Last Active:</span>
                        <strong style="color: var(--color-primary);">${lastSeen}</strong>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function renderEvents(events) {
    const tbody = document.getElementById("events-table-body");
    if (!tbody) return;
    
    if (events.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; color: var(--text-dim); padding: 30px;">
                    No Entry/Exit events recorded in this session.
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = events.map(e => {
        const timeStr = new Date(e.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const isEntry = e.event_type === "entry";
        const badgeClass = isEntry ? "entry" : "exit";
        const badgeLabel = isEntry ? "↓ ENTRY" : "↑ EXIT";
        const duration = e.session_duration_sec > 0 ? `${e.session_duration_sec.toFixed(1)}s` : "-";
        const imgPath = e.image_path ? `/${e.image_path}` : "";
        
        return `
            <tr>
                <td><strong style="color: #fff;">#${e.event_id}</strong></td>
                <td>
                    ${imgPath ? `<img src="${imgPath}" class="event-thumbnail" onclick="openLightbox('${imgPath}', '${e.visitor_id} (${badgeLabel})')" title="Click to enlarge">` : `<span style="color: var(--text-dim);">-</span>`}
                </td>
                <td><strong style="color: var(--color-accent);">${e.visitor_id}</strong></td>
                <td><span class="event-badge ${badgeClass}">${badgeLabel}</span></td>
                <td>${timeStr}</td>
                <td><span style="color: var(--color-warning);">${duration}</span></td>
                <td><span style="color: var(--text-dim);">Frame #${e.frame_number}</span></td>
            </tr>
        `;
    }).join('');
}

function renderTerminalLogs(lines) {
    const container = document.getElementById("terminal-logs-content");
    if (!container) return;
    
    container.innerHTML = lines.map(line => {
        let cls = "";
        if (line.includes("[EVENT:ENTRY]")) cls = "entry-log";
        else if (line.includes("[EVENT:EXIT]")) cls = "exit-log";
        else if (line.includes("[FACE:REGISTER]")) cls = "register-log";
        else if (line.includes("[FACE:RECOGNIZE]")) cls = "recognize-log";
        
        return `<div class="terminal-line ${cls}">${escapeHtml(line)}</div>`;
    }).join('');
    
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

/* Video & Config Management */
async function fetchVideosList() {
    try {
        const res = await fetch("/api/videos");
        if (!res.ok) return;
        const data = await res.json();
        state.availableVideos = data.videos || [];
        
        const select = document.getElementById("video-source-select");
        if (!select) return;
        
        select.innerHTML = state.availableVideos.map(v => {
            const label = v.split("/").pop();
            return `<option value="${v}">${label}</option>`;
        }).join('');
        
        select.onchange = async () => {
            const selected = select.value;
            updateCountersDirect(0, 0, 0, 0);
            await fetch("/api/stream/switch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source: selected, is_rtsp: false })
            });
            state.lastVisitorsHash = "";
            state.lastEventsHash = "";
            refreshVideoStream();
            pollLiveState();
        };
    } catch (e) {
        console.error("Videos list error:", e);
    }
}

async function fetchConfig() {
    try {
        const res = await fetch("/api/config");
        if (!res.ok) return;
        state.config = await res.json();
        
        const skip = state.config.detection?.frame_skip ?? 0;
        const sim = state.config.recognition?.similarity_threshold ?? 0.50;
        const conf = state.config.detection?.confidence_threshold ?? 0.40;
        
        const skipEl = document.getElementById("cfg-frame-skip");
        const simEl = document.getElementById("cfg-sim-thresh");
        const confEl = document.getElementById("cfg-conf-thresh");
        
        if (skipEl) skipEl.value = skip;
        if (simEl) simEl.value = sim;
        if (confEl) confEl.value = conf;
        
        const skipValEl = document.getElementById("cfg-frame-skip-val");
        if (skipValEl) skipValEl.textContent = skip;
    } catch (e) {
        console.error("Config fetch error:", e);
    }
}

function refreshVideoStream() {
    const img = document.getElementById("main-video-feed");
    if (img) {
        img.src = `/video_feed?t=${Date.now()}`;
    }
}

/* Lightweight Canvas Real-Time Chart */
function initChart() {
    const canvas = document.getElementById("telemetry-chart");
    if (!canvas) return;
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = 160;
}

function pushChartData(label, cpu, fps, active) {
    const maxPoints = 20;
    state.chartData.labels.push(label);
    state.chartData.cpu.push(cpu);
    state.chartData.fps.push(fps);
    state.chartData.active.push(active);
    
    if (state.chartData.labels.length > maxPoints) {
        state.chartData.labels.shift();
        state.chartData.cpu.shift();
        state.chartData.fps.shift();
        state.chartData.active.shift();
    }
    
    drawChart();
}

function drawChart() {
    const canvas = document.getElementById("telemetry-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    const w = canvas.width;
    const h = canvas.height;
    
    ctx.clearRect(0, 0, w, h);
    
    // Background grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let y = 20; y < h; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }
    
    const pts = state.chartData.labels.length;
    if (pts < 2) return;
    
    const stepX = w / (pts - 1);
    
    // Draw CPU Line (%: 0-100)
    ctx.strokeStyle = "#38bdf8"; // Sky blue
    ctx.lineWidth = 2;
    ctx.beginPath();
    state.chartData.cpu.forEach((val, i) => {
        const x = i * stepX;
        const y = h - (val / 100) * (h - 20) - 10;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    
    // Draw FPS Line (0-60)
    ctx.strokeStyle = "#00e673"; // Emerald green
    ctx.lineWidth = 2;
    ctx.beginPath();
    state.chartData.fps.forEach((val, i) => {
        const x = i * stepX;
        const y = h - (Math.min(60, val) / 60) * (h - 20) - 10;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    
    // Legend
    ctx.font = "10px Inter, sans-serif";
    ctx.fillStyle = "#38bdf8";
    ctx.fillText("■ CPU %", 10, 14);
    ctx.fillStyle = "#00e673";
    ctx.fillText("■ FPS", 70, 14);
}

function refreshVideoStream() {
    const img = document.getElementById("main-video-feed");
    if (img) {
        img.src = `/video_feed?t=${Date.now()}`;
    }
}

async function fetchConfig() {
    try {
        const res = await fetch("/api/config");
        if (!res.ok) return;
        state.config = await res.json();
        
        const det = state.config.detection || {};
        const rec = state.config.recognition || {};
        
        const skipInput = document.getElementById("cfg-frame-skip");
        const simInput = document.getElementById("cfg-sim-thresh");
        const confInput = document.getElementById("cfg-conf-thresh");
        const skipVal = document.getElementById("cfg-frame-skip-val");
        
        if (skipInput && det.frame_skip !== undefined) skipInput.value = det.frame_skip;
        if (simInput && rec.similarity_threshold !== undefined) simInput.value = rec.similarity_threshold;
        if (confInput && det.confidence_threshold !== undefined) confInput.value = det.confidence_threshold;
        if (skipVal && det.frame_skip !== undefined) skipVal.textContent = det.frame_skip;
    } catch (e) {
        console.error("Failed to load config:", e);
    }
}

async function fetchVideosList() {
    try {
        const res = await fetch("/api/videos");
        if (!res.ok) return;
        const data = await res.json();
        state.availableVideos = data.videos || [];
        
        const select = document.getElementById("video-source-select");
        if (!select) return;
        
        let activeSource = "";
        try {
            const cfgRes = await fetch("/api/config");
            if (cfgRes.ok) {
                const cfg = await cfgRes.json();
                activeSource = (cfg.stream && cfg.stream.source) || "";
            }
        } catch (_) {}
        
        select.innerHTML = "";
        state.availableVideos.forEach(v => {
            const opt = document.createElement("option");
            opt.value = v;
            const filename = v.split("/").pop();
            opt.textContent = filename;
            if (v === activeSource || filename === activeSource || activeSource.endsWith(filename)) {
                opt.selected = true;
            }
            select.appendChild(opt);
        });
        
        select.onchange = async () => {
            const selectedVideo = select.value;
            if (!selectedVideo) return;
            
            updateCountersDirect(0, 0, 0, 0);
            
            await fetch("/api/stream/switch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source: selectedVideo, is_rtsp: false })
            });
            
            state.lastVisitorsHash = "";
            state.lastEventsHash = "";
            state.lastLogsHash = "";
            refreshVideoStream();
            pollLiveState();
        };
    } catch (e) {
        console.error("Failed to fetch videos list:", e);
    }
}

