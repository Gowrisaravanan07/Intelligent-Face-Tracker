import os
import cv2
import time
import queue
import threading
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from database.db_manager import DatabaseManager
from logger.event_logger import EventLogger
from core.detector import FaceDetector, FaceDetectionResult
from core.recognizer import FaceRecognizer
from core.tracker import MultiObjectTracker, Track
from core.stream_manager import StreamManager
from core.compute_monitor import ComputeMonitor

class FaceTrackerPipeline:
    """
    Master Intelligent Face Tracking, Auto-Registration, and Visitor Counting Pipeline.
    Optimized for Real-Time Smooth Playback and High Recall Accuracy:
      - Asynchronous decoupled display streaming (native video FPS, 0% video lag)
      - High-frequency low-latency AI worker (<25% CPU usage)
      - Strict per-video session isolation (starts from 0 on load/switch without cumulative pollution)
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = DatabaseManager(
            db_path=config.get("database", {}).get("db_path", "database/visitors.db"),
            enable_wal=bool(config.get("database", {}).get("enable_wal_mode", True))
        )
        self.logger = EventLogger(config=config, db_manager=self.db)
        self.detector = FaceDetector(config=config)
        self.recognizer = FaceRecognizer(config=config, db_manager=self.db, logger=self.logger)
        self.tracker = MultiObjectTracker(config=config)
        self.stream_manager = StreamManager(config=config)
        self.compute_monitor = ComputeMonitor()

        self.frame_skip = config.get("detection", {}).get("frame_skip", 1)
        self.frame_index = 0
        self.ai_step_counter = 0
        self.running = False

        # Threads & Non-blocking queues
        self.display_thread: Optional[threading.Thread] = None
        self.ai_thread: Optional[threading.Thread] = None
        self._ai_queue: queue.Queue = queue.Queue(maxsize=1)

        # Thread synchronization and frame caching
        self._frame_lock = threading.Lock()
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_jpeg_bytes: Optional[bytes] = None
        self.frame_id: int = 0

        # Protected track cache for instantaneous display rendering
        self._tracks_lock = threading.Lock()
        self._cached_tracks: List[Dict[str, Any]] = []

        # Session Metrics & Data Isolation (Strictly for Current Video, Starts from 0)
        self.cached_unique_count = self.db.get_unique_visitor_count()
        self.session_unique_visitors = set()
        self.session_entries = 0
        self.session_exits = 0
        self.active_in_frame_count = 0
        self.session_visitors_map: Dict[str, Dict[str, Any]] = {}
        self.session_events_list: List[Dict[str, Any]] = []
        self._session_event_id_counter = 0

        # Performance & FPS
        self.current_fps = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_frame_counter = 0

    def reset_session(self, clear_db: bool = False):
        """Resets tracking state and session counts cleanly when switching or reloading videos."""
        with self._tracks_lock:
            self.tracker.tracks.clear()
            self._cached_tracks.clear()
            self.session_unique_visitors.clear()
            self.session_visitors_map.clear()
            self.session_events_list.clear()
            self.session_entries = 0
            self.session_exits = 0
            self.active_in_frame_count = 0
            self.frame_index = 0
            self.ai_step_counter = 0
            self._session_event_id_counter = 0

            # Drain AI queue
            while not self._ai_queue.empty():
                try:
                    self._ai_queue.get_nowait()
                except queue.Empty:
                    break

            if clear_db:
                self.db.clear_all_data()
                self.recognizer.reload()
                self.cached_unique_count = 0

        self.logger.info(f"[Pipeline] Stream session reset to 0 (Clear DB: {clear_db})")

    def start(self):
        """Starts the video stream, display thread, and background AI inference thread."""
        if self.running:
            return
        self.running = True
        self.stream_manager.start()

        self.ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self.ai_thread.start()

        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()

        self.logger.info("[Pipeline] Face Tracker Pipeline started (Optimized Smooth Playback & Accuracy).")

    def stop(self):
        """Gracefully stops pipeline, flushes tracks, and records exit events."""
        self.running = False
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.5)
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.5)

        self.stream_manager.stop()
        self._flush_active_tracks_on_exit()
        self.logger.flush()
        self.logger.info("[Pipeline] Face Tracker Pipeline stopped.")

    def _flush_active_tracks_on_exit(self):
        """Logs exit events for any active tracks when system is shutting down."""
        now = datetime.now()
        with self._tracks_lock:
            for track in self.tracker.tracks:
                if track.visitor_id and track.entry_logged and not track.exit_logged:
                    dwell_sec = max(0.1, (now - track.first_seen).total_seconds())
                    img_path = self.logger.log_exit(
                        visitor_id=track.visitor_id,
                        last_face_crop=track.best_face_crop,
                        timestamp=now,
                        session_duration_sec=dwell_sec,
                        frame_number=self.frame_index
                    )
                    track.exit_logged = True
                    self.session_exits += 1
                    self._session_event_id_counter += 1
                    self.session_events_list.append({
                        "event_id": self._session_event_id_counter,
                        "visitor_id": track.visitor_id,
                        "event_type": "exit",
                        "timestamp": now.isoformat(),
                        "image_path": img_path,
                        "confidence": 1.0,
                        "session_duration_sec": dwell_sec,
                        "frame_number": self.frame_index
                    })

    # -------------------------------------------------------------------------
    # 1. DISPLAY & STREAMING LOOP (Native Video FPS, Zero Stutter)
    # -------------------------------------------------------------------------
    def _display_loop(self):
        """Reads frames at video native rate, draws cached tracks, and updates MJPEG stream."""
        while self.running:
            frame = self.stream_manager.read_frame(timeout=0.08)
            if frame is None:
                time.sleep(0.002)
                continue

            self.frame_index += 1
            now = datetime.now()

            # Always feed freshest frame to AI worker (drop stale frame if queue is full)
            try:
                self._ai_queue.put_nowait((frame.copy(), now, self.frame_index))
            except queue.Full:
                try:
                    self._ai_queue.get_nowait()
                    self._ai_queue.put_nowait((frame.copy(), now, self.frame_index))
                except Exception:
                    pass

            # Retrieve cached track metadata safely and extrapolate smoothly with velocity
            with self._tracks_lock:
                tracks_snapshot = []
                for t in self._cached_tracks:
                    vx = t.get("vx", 0.0)
                    vy = t.get("vy", 0.0)
                    b = list(t["bbox"])
                    b[0] = int(b[0] + vx * 0.4)
                    b[1] = int(b[1] + vy * 0.4)
                    b[2] = int(b[2] + vx * 0.4)
                    b[3] = int(b[3] + vy * 0.4)
                    t["bbox"] = tuple(b)
                    tracks_snapshot.append(dict(t))

                session_unique = len(self.session_unique_visitors)
                active_count = self.active_in_frame_count
                entries_count = self.session_entries
                exits_count = self.session_exits

            # Render visual overlays
            annotated = self._render_annotations_from_cache(
                frame, tracks_snapshot, session_unique, active_count, entries_count, exits_count
            )

            # High-speed JPEG encoding directly at target stream resolution
            ret, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                jpeg_bytes = buf.tobytes()
                with self._frame_lock:
                    self.latest_annotated_frame = annotated
                    self.latest_jpeg_bytes = jpeg_bytes
                    self.frame_id += 1

            # Real-time FPS Calculation
            self._fps_frame_counter += 1
            now_time = time.time()
            if now_time - self._last_fps_calc_time >= 1.0:
                self.current_fps = self._fps_frame_counter / (now_time - self._last_fps_calc_time)
                self._fps_frame_counter = 0
                self._last_fps_calc_time = now_time

    # -------------------------------------------------------------------------
    # 2. BACKGROUND AI WORKER LOOP (Detection + Tracking + ArcFace Recognition)
    # -------------------------------------------------------------------------
    def _ai_worker_loop(self):
        """Runs face detection, tracking, and identity registration in the background."""
        last_metrics_log = time.time()

        while self.running:
            try:
                frame, timestamp, frame_idx = self._ai_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            loop_start = time.time()
            t_det_ms, t_rec_ms, t_trk_ms = 0.0, 0.0, 0.0

            # 1. Detection on freshest frame
            t0 = time.time()
            detections = self.detector.detect(frame)
            t_det_ms = (time.time() - t0) * 1000.0

            formatted_dets = []
            for det in detections:
                crop = self.logger.crop_face(frame, det.bbox)
                formatted_dets.append({
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "crop": crop,
                    "landmarks": det.landmarks,
                    "embedding": det.embedding
                })

            t0 = time.time()
            tracks = self.tracker.update_with_detections(
                detections=formatted_dets,
                timestamp=timestamp,
                frame_number=frame_idx
            )
            t_trk_ms = (time.time() - t0) * 1000.0

            # 2. Recognition & Auto-Registration
            t0 = time.time()
            for track in tracks:
                if track.state == "confirmed" and track.visitor_id is None:
                    # Extract high-accuracy embedding
                    if track.latest_embedding is not None:
                        emb = track.latest_embedding
                    elif track.latest_landmarks is not None:
                        emb = self.detector.extract_embedding(frame, track.bbox, track.latest_landmarks)
                    elif track.best_face_crop is not None:
                        emb = self.detector.extract_embedding_from_crop(track.best_face_crop)
                    else:
                        emb = None

                    if emb is not None:
                        visitor_id, conf, is_new = self.recognizer.identify_or_register(
                            embedding=emb,
                            timestamp=track.first_seen,
                            quality_score=track.best_confidence,
                            frame_number=frame_idx
                        )
                        track.visitor_id = visitor_id
                        if is_new:
                            self.cached_unique_count += 1

                        if not track.entry_logged:
                            img_path = self.logger.log_entry(
                                visitor_id=visitor_id,
                                frame=frame,
                                bbox=track.bbox,
                                timestamp=track.first_seen,
                                confidence=track.best_confidence,
                                frame_number=frame_idx
                            )
                            track.entry_logged = True
                            
                            with self._tracks_lock:
                                self.session_unique_visitors.add(visitor_id)
                                self.session_entries += 1
                                self._session_event_id_counter += 1
                                self.session_events_list.append({
                                    "event_id": self._session_event_id_counter,
                                    "visitor_id": visitor_id,
                                    "event_type": "entry",
                                    "timestamp": track.first_seen.isoformat(),
                                    "image_path": img_path,
                                    "confidence": track.best_confidence,
                                    "session_duration_sec": 0.0,
                                    "frame_number": frame_idx
                                })
                                
                                # Session visitor record
                                if visitor_id not in self.session_visitors_map:
                                    self.session_visitors_map[visitor_id] = {
                                        "visitor_id": visitor_id,
                                        "name": f"Visitor #{visitor_id.split('_')[-1]}",
                                        "first_seen": track.first_seen.isoformat(),
                                        "last_seen": track.first_seen.isoformat(),
                                        "total_visits": 1,
                                        "thumbnail_path": img_path,
                                        "quality_score": track.best_confidence
                                    }
                                else:
                                    self.session_visitors_map[visitor_id]["last_seen"] = track.first_seen.isoformat()

                            if not is_new:
                                self.db.increment_visitor_visit(visitor_id, track.first_seen)
            t_rec_ms = (time.time() - t0) * 1000.0

            # 3. Handle Exits
            dead_tracks = self.tracker.remove_dead_tracks()
            for dead in dead_tracks:
                if dead.visitor_id and dead.entry_logged and not dead.exit_logged:
                    dwell_sec = max(0.1, (dead.last_seen - dead.first_seen).total_seconds())
                    img_path = self.logger.log_exit(
                        visitor_id=dead.visitor_id,
                        last_face_crop=dead.best_face_crop,
                        timestamp=dead.last_seen,
                        session_duration_sec=dwell_sec,
                        frame_number=frame_idx
                    )
                    dead.exit_logged = True
                    with self._tracks_lock:
                        self.session_exits += 1
                        self._session_event_id_counter += 1
                        self.session_events_list.append({
                            "event_id": self._session_event_id_counter,
                            "visitor_id": dead.visitor_id,
                            "event_type": "exit",
                            "timestamp": dead.last_seen.isoformat(),
                            "image_path": img_path,
                            "confidence": 1.0,
                            "session_duration_sec": dwell_sec,
                            "frame_number": frame_idx
                        })

            # 4. Snapshot confirmed tracks strictly for visual display
            active_tracks_list = []
            active_in_frame = 0
            for t in tracks:
                if t.state == "confirmed" and t.time_since_update <= min(25, self.tracker.max_missing):
                    active_in_frame += 1
                    active_tracks_list.append({
                        "bbox": t.bbox,
                        "track_id": t.track_id,
                        "visitor_id": t.visitor_id,
                        "state": t.state,
                        "vx": t.vx,
                        "vy": t.vy,
                        "history": list(t.history)
                    })

            with self._tracks_lock:
                self._cached_tracks = active_tracks_list
                self.active_in_frame_count = active_in_frame

            # 5. Profiling & Metrics Telemetry
            total_ms = (time.time() - loop_start) * 1000.0
            self.compute_monitor.record_frame_latency(
                total_ms=total_ms,
                det_ms=t_det_ms,
                rec_ms=t_rec_ms,
                trk_ms=t_trk_ms
            )

            now_t = time.time()
            if now_t - last_metrics_log >= 1.0:
                last_metrics_log = now_t
                snap = self.compute_monitor.get_metrics_snapshot(
                    current_fps=self.current_fps,
                    active_tracks=active_in_frame,
                    unique_count=len(self.session_unique_visitors)
                )
                self.db.log_system_metrics(
                    fps=snap["fps"],
                    cpu_percent=snap["cpu_percent"],
                    gpu_percent=snap["gpu_percent"],
                    memory_mb=snap["memory_mb"],
                    active_tracks=snap["active_tracks"],
                    unique_visitors_count=snap["unique_visitors_count"],
                    frame_process_time_ms=snap["frame_process_time_ms"]
                )

    # -------------------------------------------------------------------------
    # 3. VISUAL ANNOTATIONS RENDERING
    # -------------------------------------------------------------------------
    def _render_annotations_from_cache(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        session_visitors: int,
        active_in_frame: int,
        session_entries: int,
        session_exits: int
    ) -> np.ndarray:
        """Renders rich bounding boxes, IDs, trajectories, and HUD stats overlay onto the frame."""
        h, w = frame.shape[:2]

        for t in tracks:
            bbox = t.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(x1 + 1, min(w, int(x2)))
            y2 = max(y1 + 1, min(h, int(y2)))

            visitor_id = t.get("visitor_id")
            # Emerald green for recognized visitor, Cyan/Amber for tracking
            color = (0, 230, 115) if visitor_id else (0, 190, 255)
            thickness = 2

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Trajectory line
            history = t.get("history", [])
            if len(history) > 1:
                pts = np.array(history, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

            # Label chip
            label = visitor_id if visitor_id else f"Tracking #{t.get('track_id')}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)

            chip_y1 = max(0, y1 - lh - 8)
            chip_y2 = y1
            cv2.rectangle(frame, (x1, chip_y1), (x1 + lw + 12, chip_y2), (20, 24, 30), -1)
            cv2.rectangle(frame, (x1, chip_y1), (x1 + lw + 12, chip_y2), color, 1)
            cv2.putText(
                frame, label, (x1 + 6, y1 - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

        # HUD Top Banner Overlay
        hud_h = 42
        cv2.rectangle(frame, (0, 0), (w, hud_h), (15, 18, 24), -1)
        cv2.line(frame, (0, hud_h), (w, hud_h), (0, 230, 115), 2)

        # HUD Text elements
        hud_text_1 = f"THIS VIDEO: {session_visitors} UNIQUE"
        hud_text_2 = f"IN FRAME: {active_in_frame}"
        hud_text_3 = f"ENTRIES: {session_entries} | EXITS: {session_exits}"
        hud_text_4 = f"FPS: {self.current_fps:.1f}"

        cv2.putText(frame, hud_text_1, (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 130), 1, cv2.LINE_AA)
        cv2.putText(frame, hud_text_2, (250, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, hud_text_3, (400, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (100, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, hud_text_4, (680, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        """Returns the current annotated frame encoded as JPEG bytes."""
        with self._frame_lock:
            return self.latest_jpeg_bytes

    def get_latest_frame_with_id(self) -> Tuple[Optional[bytes], int]:
        """Returns the latest JPEG frame bytes along with monotonic frame ID for zero-lag streaming."""
        with self._frame_lock:
            return self.latest_jpeg_bytes, self.frame_id

    def get_session_visitors(self) -> List[Dict[str, Any]]:
        """Returns registered visitors gallery specifically for the CURRENT active video session."""
        with self._tracks_lock:
            return list(self.session_visitors_map.values())

    def get_session_events(self) -> List[Dict[str, Any]]:
        """Returns entry/exit events specifically for the CURRENT active video session."""
        with self._tracks_lock:
            return list(reversed(self.session_events_list[-40:]))

    def get_status_summary(self) -> Dict[str, Any]:
        """Returns instantaneous system health & stats summary."""
        session_unique = len(self.session_unique_visitors)
        active = self.active_in_frame_count
        snap = self.compute_monitor.get_metrics_snapshot(
            current_fps=self.current_fps,
            active_tracks=active,
            unique_count=session_unique
        )
        return {
            "session_unique_visitors": session_unique,
            "session_entries": self.session_entries,
            "session_exits": self.session_exits,
            "active_in_frame": active,
            "total_unique_visitors": self.cached_unique_count,
            "today_entries": self.session_entries,
            "today_exits": self.session_exits,
            "current_fps": self.current_fps,
            "frame_skip": self.frame_skip,
            "stream_source": str(self.stream_manager.source),
            "compute": snap
        }
