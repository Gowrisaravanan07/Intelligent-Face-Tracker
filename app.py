import os
import sys

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

import json
import time
import csv
import io
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List

import cv2
from fastapi import FastAPI, Request, Query, Body, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.pipeline import FaceTrackerPipeline
from database.db_manager import DatabaseManager

# Load base configuration
CONFIG_PATH = "config.json"
def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

config = load_config()

app = FastAPI(
    title="Katomaran AI - Face Tracker & Visitor Counting Hub",
    description="Real-Time AI Face Tracking, Auto-Registration, and Unique Visitor Counting System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = FaceTrackerPipeline(config=config)
pipeline.start()

# Mount static files and logs directory for images
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("logs/entries", exist_ok=True)
os.makedirs("logs/exits", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/logs", StaticFiles(directory="logs"), name="logs")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serves dashboard single-page application."""
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard Loading...</h1>")


async def gen_frames():
    """
    High-Performance Zero-Lag MJPEG Stream Generator.
    Yields each freshly rendered frame to the client without queue backlog or buffering lag.
    """
    last_frame_id = -1
    try:
        while True:
            frame_bytes, frame_id = pipeline.get_latest_frame_with_id()
            if frame_bytes is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            await asyncio.sleep(0.002)
    except (asyncio.CancelledError, ConnectionResetError):
        pass


@app.get("/video_feed")
async def video_feed():
    """MJPEG Live Video Streaming Endpoint."""
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/health")
async def health_check():
    """Returns application health status."""
    return JSONResponse({"status": "healthy", "pipeline_running": pipeline.running, "fps": pipeline.current_fps})


@app.get("/api/stats")
async def get_stats():
    """Returns top KPI metrics and system summary."""
    summary = pipeline.get_status_summary()
    return JSONResponse(summary)


@app.get("/api/visitors")
async def get_visitors(session_only: bool = True, limit: int = 100, offset: int = 0):
    """Returns registered visitors gallery (session-isolated by default)."""
    if session_only:
        visitors = pipeline.get_session_visitors()
    else:
        visitors = pipeline.db.get_visitors_gallery(limit=limit, offset=offset)
    return JSONResponse({"total": len(visitors), "visitors": visitors})


@app.get("/api/events")
async def get_events(session_only: bool = True, limit: int = 50, event_type: Optional[str] = None):
    """Returns recent entry/exit events (session-isolated by default)."""
    if session_only:
        events = pipeline.get_session_events()
    else:
        events = pipeline.db.get_recent_events(limit=limit, event_type=event_type)
    return JSONResponse({"total": len(events), "events": events})


@app.get("/api/compute")
async def get_compute_telemetry(limit: int = 60):
    """Returns compute telemetry time-series for CPU/GPU chart."""
    metrics = pipeline.db.get_recent_metrics(limit=limit)
    return JSONResponse({"metrics": metrics})


def _fetch_live_state_sync() -> Dict[str, Any]:
    summary = pipeline.get_status_summary()
    visitors = pipeline.get_session_visitors()
    events = pipeline.get_session_events()
    log_lines = pipeline.logger.get_recent_logs(limit=30)

    return {
        "stats": summary,
        "visitors": visitors,
        "events": events,
        "logs": log_lines
    }


@app.get("/api/live_state")
async def get_live_state():
    """
    High-Speed Non-Blocking State Snapshot Endpoint.
    """
    state_data = await asyncio.to_thread(_fetch_live_state_sync)
    return JSONResponse(state_data)


@app.get("/api/logs")
async def get_events_log(lines: int = 100):
    """Returns raw content of events.log for the terminal viewer."""
    log_file = config.get("logging", {}).get("events_log_file", "logs/events.log")
    if not os.path.exists(log_file):
        return JSONResponse({"lines": ["No logs generated yet."]})

    try:
        def _read_logs():
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                all_lines = f.readlines()
                return [l.strip() for l in (all_lines[-lines:] if len(all_lines) > lines else all_lines)]
        tail = await asyncio.to_thread(_read_logs)
        return JSONResponse({"lines": tail})
    except Exception as e:
        return JSONResponse({"lines": [f"Log read error: {e}"]})


@app.get("/api/config")
async def get_current_config():
    """Returns active configuration."""
    return JSONResponse(config)


@app.post("/api/config")
async def update_config(payload: Dict[str, Any] = Body(...)):
    """Live-updates configurable parameters."""
    global config
    if "detection" in payload:
        if "frame_skip" in payload["detection"]:
            skip = int(payload["detection"]["frame_skip"])
            config["detection"]["frame_skip"] = skip
            pipeline.frame_skip = skip

        if "confidence_threshold" in payload["detection"]:
            conf = float(payload["detection"]["confidence_threshold"])
            config["detection"]["confidence_threshold"] = conf
            pipeline.detector.conf_threshold = conf

    if "recognition" in payload and "similarity_threshold" in payload["recognition"]:
        sim = float(payload["recognition"]["similarity_threshold"])
        config["recognition"]["similarity_threshold"] = sim
        pipeline.recognizer.similarity_threshold = sim

    # Save to disk
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    pipeline.logger.info(f"[Config] Updated live parameters: {payload}")
    return JSONResponse({"status": "success", "config": config})


@app.post("/api/stream/switch")
async def switch_stream(payload: Dict[str, Any] = Body(...)):
    """Dynamically switches input video file or RTSP URL asynchronously without blocking server."""
    source = payload.get("source")
    is_rtsp = payload.get("is_rtsp", False)

    if not source:
        raise HTTPException(status_code=400, detail="Missing stream source.")

    config["stream"]["source"] = source
    config["stream"]["is_rtsp"] = is_rtsp

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Reset video-specific session tracks and counters
    pipeline.reset_session(clear_db=False)

    # Perform capture reset asynchronously in thread pool
    await asyncio.to_thread(pipeline.stream_manager.set_source, source, is_rtsp)
    pipeline.logger.info(f"[Stream] Switched stream source to {source} (RTSP: {is_rtsp})")
    return JSONResponse({"status": "success", "new_source": source, "is_rtsp": is_rtsp})


@app.post("/api/session/reset")
async def reset_session(payload: Dict[str, Any] = Body(default={})):
    """Resets tracking counts and optionally clears all historical database records."""
    clear_db = bool(payload.get("clear_db", False))
    pipeline.reset_session(clear_db=clear_db)
    return JSONResponse({"status": "success", "cleared_db": clear_db})


@app.get("/api/videos")
async def list_available_videos():
    """Lists all sample videos in data/ folder."""
    data_dir = "data"
    videos = []
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith((".mp4", ".avi", ".mkv", ".mov")):
                videos.append(f"data/{f}")
    return JSONResponse({"videos": videos})


@app.get("/api/export/csv")
async def export_csv():
    """Exports all visitor entry/exit events as CSV download."""
    events = pipeline.db.get_recent_events(limit=5000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Event ID", "Visitor ID", "Event Type", "Timestamp",
        "Confidence", "Dwell Duration (s)", "Frame Number", "Image Path"
    ])
    for e in events:
        writer.writerow([
            e.get("event_id"),
            e.get("visitor_id"),
            e.get("event_type"),
            e.get("timestamp"),
            e.get("confidence"),
            e.get("session_duration_sec"),
            e.get("frame_number"),
            e.get("image_path")
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=visitor_events_export.csv"}
    )


def run_server():
    """Starts FastAPI Uvicorn Server."""
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 8000)
    print(f"\n[Dashboard] Starting Katomaran AI Visitor Intelligence Web UI on http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_server()
