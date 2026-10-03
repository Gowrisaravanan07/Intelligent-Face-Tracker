-- ====================================================================
-- Katomaran Hackathon: Intelligent Face Tracker & Visitor Counter
-- Database Schema for Visitors, Events, and Performance Metrics
-- ====================================================================

-- 1. Registered Visitors Table
CREATE TABLE IF NOT EXISTS visitors (
    visitor_id TEXT PRIMARY KEY,
    name TEXT DEFAULT 'Visitor',
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    total_visits INTEGER DEFAULT 1,
    embedding_blob BLOB,
    embedding_dim INTEGER DEFAULT 512,
    thumbnail_path TEXT,
    quality_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Entry and Exit Events Table
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_id TEXT NOT NULL,
    event_type TEXT CHECK(event_type IN ('entry', 'exit')),
    timestamp TIMESTAMP NOT NULL,
    image_path TEXT,
    confidence REAL DEFAULT 1.0,
    session_duration_sec REAL DEFAULT 0.0,
    frame_number INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (visitor_id) REFERENCES visitors (visitor_id) ON DELETE CASCADE
);

-- 3. System Compute & Hardware Telemetry Table
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fps REAL,
    cpu_percent REAL,
    gpu_percent REAL,
    memory_mb REAL,
    active_tracks INTEGER,
    unique_visitors_count INTEGER,
    frame_process_time_ms REAL
);

-- Indexes for lightning fast queries
CREATE INDEX IF NOT EXISTS idx_visitors_last_seen ON visitors(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_events_visitor_id ON events(visitor_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp DESC);
