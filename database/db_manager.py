import os
import sqlite3
import threading
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

class DatabaseManager:
    """
    Thread-safe SQLite Database Manager for Visitor Auto-Registration,
    Event Logging (Entry/Exit), and System Metrics Tracking.
    """
    def __init__(self, db_path: str = "database/visitors.db", enable_wal: bool = True):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.enable_wal = enable_wal
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            if self.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.connection = conn
        return self._local.connection

    def _init_db(self):
        """Initializes database schema and indexes."""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        conn = self._get_connection()
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
        else:
            conn.execute("""
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
            """)
            conn.execute("""
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
                    FOREIGN KEY (visitor_id) REFERENCES visitors (visitor_id)
                );
            """)
            conn.execute("""
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
            """)
        conn.commit()

    def register_visitor(
        self,
        visitor_id: str,
        embedding: np.ndarray,
        first_seen: datetime,
        thumbnail_path: Optional[str] = None,
        quality_score: float = 1.0,
        name: Optional[str] = None
    ) -> bool:
        """
        Registers a newly discovered unique visitor in the database.
        """
        conn = self._get_connection()
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-6)
        blob = norm_emb.astype(np.float32).tobytes()
        display_name = name or f"Visitor #{visitor_id.split('_')[-1]}"

        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO visitors (
                        visitor_id, name, first_seen, last_seen, total_visits,
                        embedding_blob, embedding_dim, thumbnail_path, quality_score
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                    """,
                    (
                        visitor_id,
                        display_name,
                        first_seen.isoformat(),
                        first_seen.isoformat(),
                        blob,
                        len(embedding),
                        thumbnail_path,
                        quality_score
                    )
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def update_visitor_embedding(
        self,
        visitor_id: str,
        new_embedding: np.ndarray,
        alpha: float = 0.15,
        new_thumbnail: Optional[str] = None
    ) -> bool:
        """
        Updates the visitor's stored facial embedding using Exponential Moving Average (EMA).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT embedding_blob, embedding_dim FROM visitors WHERE visitor_id = ?", (visitor_id,))
        row = cursor.fetchone()
        if not row or not row["embedding_blob"]:
            return False

        old_emb = np.frombuffer(row["embedding_blob"], dtype=np.float32)
        norm_new = new_embedding / (np.linalg.norm(new_embedding) + 1e-6)
        updated_emb = (1.0 - alpha) * old_emb + alpha * norm_new
        updated_emb = updated_emb / (np.linalg.norm(updated_emb) + 1e-6)
        blob = updated_emb.astype(np.float32).tobytes()

        with conn:
            if new_thumbnail:
                conn.execute(
                    "UPDATE visitors SET embedding_blob = ?, thumbnail_path = ?, last_seen = ? WHERE visitor_id = ?",
                    (blob, new_thumbnail, datetime.now().isoformat(), visitor_id)
                )
            else:
                conn.execute(
                    "UPDATE visitors SET embedding_blob = ?, last_seen = ? WHERE visitor_id = ?",
                    (blob, datetime.now().isoformat(), visitor_id)
                )
        return True

    def increment_visitor_visit(self, visitor_id: str, timestamp: datetime) -> None:
        """Increments total visit count when an existing visitor re-enters on a new session."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                "UPDATE visitors SET total_visits = total_visits + 1, last_seen = ? WHERE visitor_id = ?",
                (timestamp.isoformat(), visitor_id)
            )

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Retrieves all registered visitor IDs and their corresponding normalized embedding matrix.
        Returns (visitor_ids, embeddings_matrix) where matrix shape is (N, 512).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT visitor_id, embedding_blob, embedding_dim FROM visitors")
        rows = cursor.fetchall()

        ids = []
        vectors = []
        for r in rows:
            if r["embedding_blob"]:
                vec = np.frombuffer(r["embedding_blob"], dtype=np.float32)
                ids.append(r["visitor_id"])
                vectors.append(vec)

        if len(vectors) == 0:
            return [], np.empty((0, 512), dtype=np.float32)

        return ids, np.vstack(vectors)

    def log_event(
        self,
        visitor_id: str,
        event_type: str,
        timestamp: datetime,
        image_path: Optional[str] = None,
        confidence: float = 1.0,
        session_duration_sec: float = 0.0,
        frame_number: int = 0
    ) -> int:
        """
        Logs a single entry or exit event to the events table.
        """
        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    visitor_id, event_type, timestamp, image_path,
                    confidence, session_duration_sec, frame_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    visitor_id,
                    event_type,
                    timestamp.isoformat(),
                    image_path,
                    confidence,
                    session_duration_sec,
                    frame_number
                )
            )
            # Update last_seen in visitors table
            conn.execute(
                "UPDATE visitors SET last_seen = ? WHERE visitor_id = ?",
                (timestamp.isoformat(), visitor_id)
            )
            return cursor.lastrowid

    def get_unique_visitor_count(self) -> int:
        """Returns total count of registered unique visitors."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM visitors")
        row = cursor.fetchone()
        return row["count"] if row else 0

    def get_today_event_counts(self) -> Dict[str, int]:
        """Returns count of entries and exits recorded today."""
        conn = self._get_connection()
        cursor = conn.cursor()
        today_start = datetime.now().strftime("%Y-%m-%d 00:00:00")
        cursor.execute(
            """
            SELECT event_type, COUNT(*) as count
            FROM events
            WHERE timestamp >= ?
            GROUP BY event_type
            """,
            (today_start,)
        )
        counts = {"entry": 0, "exit": 0}
        for r in cursor.fetchall():
            counts[r["event_type"]] = r["count"]
        return counts

    def get_recent_events(self, limit: int = 50, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves recent events with joined visitor details."""
        conn = self._get_connection()
        cursor = conn.cursor()
        query = """
            SELECT e.event_id, e.visitor_id, e.event_type, e.timestamp,
                   e.image_path, e.confidence, e.session_duration_sec, e.frame_number,
                   v.name as visitor_name, v.thumbnail_path as visitor_thumbnail,
                   v.total_visits
            FROM events e
            LEFT JOIN visitors v ON e.visitor_id = v.visitor_id
        """
        params = []
        if event_type:
            query += " WHERE e.event_type = ?"
            params.append(event_type)
        query += " ORDER BY e.event_id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    def get_visitors_gallery(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieves list of registered visitors with profile info and thumbnail path."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT visitor_id, name, first_seen, last_seen, total_visits,
                   thumbnail_path, quality_score, created_at
            FROM visitors
            ORDER BY last_seen DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        return [dict(r) for r in cursor.fetchall()]

    def log_system_metrics(
        self,
        fps: float,
        cpu_percent: float,
        gpu_percent: float,
        memory_mb: float,
        active_tracks: int,
        unique_visitors_count: int,
        frame_process_time_ms: float
    ) -> None:
        """Stores a snapshot of compute performance telemetry."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO system_metrics (
                    fps, cpu_percent, gpu_percent, memory_mb,
                    active_tracks, unique_visitors_count, frame_process_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fps, cpu_percent, gpu_percent, memory_mb,
                    active_tracks, unique_visitors_count, frame_process_time_ms
                )
            )

    def get_recent_metrics(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Fetches recent compute performance metrics for real-time charting."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, fps, cpu_percent, gpu_percent, memory_mb,
                   active_tracks, unique_visitors_count, frame_process_time_ms
            FROM system_metrics
            ORDER BY id DESC LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    def clear_all_data(self) -> None:
        """Wipes visitors, events, and metrics tables for clean video testing."""
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM events;")
            conn.execute("DELETE FROM visitors;")
            conn.execute("DELETE FROM system_metrics;")
            conn.execute("VACUUM;")
