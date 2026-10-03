import os
import cv2
import queue
import logging
import threading
import collections
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

from database.db_manager import DatabaseManager

class EventLogger:
    """
    High-Throughput Zero-Lag Logging System for Hackathon Task:
    - Structured log file `events.log` for critical system events:
        * Face entry, recognition, tracking, and exit
        * Embedding generation and auto-registration
    - File system image storage:
        * `logs/entries/YYYY-MM-DD/{visitor_id}_{timestamp}_entry.jpg`
        * `logs/exits/YYYY-MM-DD/{visitor_id}_{timestamp}_exit.jpg`
    - Dual persistence to SQLite Database.
    - Asynchronous non-blocking background queue worker to eliminate disk I/O lag from AI loop.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        db_manager: Optional[DatabaseManager] = None
    ):
        self.config = config
        self.log_dir = config.get("logging", {}).get("log_dir", "logs")
        self.events_log_file = config.get("logging", {}).get("events_log_file", "logs/events.log")
        self.entries_folder = config.get("logging", {}).get("entries_folder", "logs/entries")
        self.exits_folder = config.get("logging", {}).get("exits_folder", "logs/exits")
        self.crop_padding_ratio = config.get("logging", {}).get("crop_padding_ratio", 0.25)
        self.jpeg_quality = config.get("logging", {}).get("jpeg_quality", 85)
        self.db_manager = db_manager or DatabaseManager()
        self.recent_logs = collections.deque(maxlen=100)

        # Ensure directory structure
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.entries_folder, exist_ok=True)
        os.makedirs(self.exits_folder, exist_ok=True)

        self._setup_file_logger()

        # Asynchronous background write queue
        self._async_queue = queue.Queue(maxsize=1000)
        self._worker_running = True
        self._worker_thread = threading.Thread(target=self._async_worker_loop, daemon=True)
        self._worker_thread.start()

    def _setup_file_logger(self):
        """Sets up Python logger writing formatted events to `events.log`."""
        self.logger = logging.getLogger("VisitorTracker")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Clear existing handlers to prevent duplicate logging
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter(
            fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler for events.log
        file_handler = logging.FileHandler(self.events_log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # In-memory buffer handler
        class DequeHandler(logging.Handler):
            def __init__(self, target_deque):
                super().__init__()
                self.target_deque = target_deque
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.target_deque.append(msg)
                except Exception:
                    pass

        mem_handler = DequeHandler(self.recent_logs)
        mem_handler.setLevel(logging.INFO)
        mem_handler.setFormatter(formatter)
        self.logger.addHandler(mem_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info("====================================================================")
        self.logger.info("Visitor Tracking & Face Recognition Pipeline Initialized")
        self.logger.info(f"Log File: {os.path.abspath(self.events_log_file)}")
        self.logger.info("====================================================================")

    def _async_worker_loop(self):
        """Background thread executing disk image saves and DB writes without delaying main video thread."""
        while self._worker_running or not self._async_queue.empty():
            try:
                task = self._async_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            task_type = task.get("type")
            try:
                if task_type == "save_image":
                    crop = task.get("crop")
                    path = task.get("path")
                    if crop is not None and path:
                        self.save_crop_image(crop, path)

                elif task_type == "entry_event":
                    crop = task.get("crop")
                    image_path = task.get("image_path")
                    if crop is not None and image_path:
                        self.save_crop_image(crop, image_path)

                    self.db_manager.log_event(
                        visitor_id=task["visitor_id"],
                        event_type="entry",
                        timestamp=task["timestamp"],
                        image_path=image_path,
                        confidence=task.get("confidence", 1.0),
                        session_duration_sec=0.0,
                        frame_number=task.get("frame_number", 0)
                    )

                elif task_type == "exit_event":
                    crop = task.get("crop")
                    image_path = task.get("image_path")
                    if crop is not None and image_path:
                        self.save_crop_image(crop, image_path)

                    self.db_manager.log_event(
                        visitor_id=task["visitor_id"],
                        event_type="exit",
                        timestamp=task["timestamp"],
                        image_path=image_path,
                        confidence=1.0,
                        session_duration_sec=task.get("session_duration_sec", 0.0),
                        frame_number=task.get("frame_number", 0)
                    )
            except Exception as e:
                self.logger.error(f"[AsyncWorker] Error processing log task: {e}")
            finally:
                self._async_queue.task_done()

    def _get_daily_dir(self, base_folder: str, timestamp: datetime) -> str:
        """Returns and creates daily directory path YYYY-MM-DD."""
        date_str = timestamp.strftime("%Y-%m-%d")
        daily_dir = os.path.join(base_folder, date_str)
        os.makedirs(daily_dir, exist_ok=True)
        return daily_dir

    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crops face from full frame with contextual padding.
        bbox format: (x1, y1, x2, y2)
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * self.crop_padding_ratio)
        pad_y = int(bh * self.crop_padding_ratio)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        return crop.copy()

    def save_crop_image(self, crop: np.ndarray, target_path: str) -> bool:
        """Saves cropped face image to disk with designated JPEG quality."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            cv2.imwrite(
                target_path,
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to save crop image to {target_path}: {e}")
            return False

    def log_entry(
        self,
        visitor_id: str,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        timestamp: datetime,
        confidence: float = 1.0,
        frame_number: int = 0,
        sync: bool = False
    ) -> str:
        """
        Logs ENTRY event:
        1. Crops face image
        2. Queues async saving to `logs/entries/YYYY-MM-DD/{visitor_id}_{timestamp}_entry.jpg`
        3. Appends structured line to `events.log`
        4. Inserts event into Database asynchronously (or synchronously if sync=True)
        """
        daily_dir = self._get_daily_dir(self.entries_folder, timestamp)
        time_tag = timestamp.strftime("%H%M%S_%f")[:10]
        filename = f"{visitor_id}_{time_tag}_entry.jpg"
        image_path = os.path.join(daily_dir, filename).replace("\\", "/")

        face_crop = self.crop_face(frame, bbox)

        x1, y1, x2, y2 = bbox
        self.logger.info(
            f"[EVENT:ENTRY] Visitor {visitor_id} ENTERED frame at pos ({x1},{y1},{x2},{y2}) | "
            f"Confidence: {confidence:.2f} | Frame: {frame_number} | Image: {image_path}"
        )

        if sync:
            self.save_crop_image(face_crop, image_path)
            self.db_manager.log_event(
                visitor_id=visitor_id,
                event_type="entry",
                timestamp=timestamp,
                image_path=image_path,
                confidence=confidence,
                session_duration_sec=0.0,
                frame_number=frame_number
            )
        else:
            try:
                self._async_queue.put_nowait({
                    "type": "entry_event",
                    "visitor_id": visitor_id,
                    "crop": face_crop,
                    "image_path": image_path,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "frame_number": frame_number
                })
            except queue.Full:
                self.save_crop_image(face_crop, image_path)
                self.db_manager.log_event(
                    visitor_id=visitor_id,
                    event_type="entry",
                    timestamp=timestamp,
                    image_path=image_path,
                    confidence=confidence,
                    session_duration_sec=0.0,
                    frame_number=frame_number
                )

        return image_path

    def log_exit(
        self,
        visitor_id: str,
        last_face_crop: Optional[np.ndarray],
        timestamp: datetime,
        session_duration_sec: float,
        frame_number: int = 0,
        sync: bool = False
    ) -> Optional[str]:
        """
        Logs EXIT event:
        1. Saves exit crop image to `logs/exits/YYYY-MM-DD/{visitor_id}_{timestamp}_exit.jpg`
        2. Appends structured line to `events.log`
        3. Inserts event into Database asynchronously (or synchronously if sync=True)
        """
        daily_dir = self._get_daily_dir(self.exits_folder, timestamp)
        time_tag = timestamp.strftime("%H%M%S_%f")[:10]
        filename = f"{visitor_id}_{time_tag}_exit.jpg"
        image_path = os.path.join(daily_dir, filename).replace("\\", "/") if (last_face_crop is not None and last_face_crop.size > 0) else None

        self.logger.info(
            f"[EVENT:EXIT] Visitor {visitor_id} EXITED frame | "
            f"Dwell Duration: {session_duration_sec:.2f}s | Frame: {frame_number} | "
            f"Image: {image_path or 'N/A'}"
        )

        crop_copy = last_face_crop.copy() if (last_face_crop is not None and last_face_crop.size > 0) else None

        if sync:
            if crop_copy is not None and image_path:
                self.save_crop_image(crop_copy, image_path)
            self.db_manager.log_event(
                visitor_id=visitor_id,
                event_type="exit",
                timestamp=timestamp,
                image_path=image_path,
                confidence=1.0,
                session_duration_sec=session_duration_sec,
                frame_number=frame_number
            )
        else:
            try:
                self._async_queue.put_nowait({
                    "type": "exit_event",
                    "visitor_id": visitor_id,
                    "crop": crop_copy,
                    "image_path": image_path,
                    "timestamp": timestamp,
                    "session_duration_sec": session_duration_sec,
                    "frame_number": frame_number
                })
            except queue.Full:
                if crop_copy is not None and image_path:
                    self.save_crop_image(crop_copy, image_path)
                self.db_manager.log_event(
                    visitor_id=visitor_id,
                    event_type="exit",
                    timestamp=timestamp,
                    image_path=image_path,
                    confidence=1.0,
                    session_duration_sec=session_duration_sec,
                    frame_number=frame_number
                )

        return image_path

    def log_registration(
        self,
        visitor_id: str,
        embedding_dim: int = 512,
        quality_score: float = 1.0,
        frame_number: int = 0
    ) -> None:
        """Logs automatic face registration event."""
        self.logger.info(
            f"[FACE:REGISTER] Auto-registered new visitor {visitor_id} | "
            f"Embedding: {embedding_dim}-D ArcFace | Quality: {quality_score:.2f} | Frame: {frame_number}"
        )

    def log_recognition(
        self,
        visitor_id: str,
        similarity_score: float,
        is_reidentified: bool = True,
        frame_number: int = 0
    ) -> None:
        """Logs face recognition and re-identification event."""
        if is_reidentified:
            self.logger.info(
                f"[FACE:RECOGNIZE] Re-identified visitor {visitor_id} (Cosine Sim: {similarity_score:.4f}) | "
                f"Count not incremented | Frame: {frame_number}"
            )
        else:
            self.logger.info(
                f"[FACE:SEARCH] No matching identity found (Max Sim: {similarity_score:.4f}) -> Generating new registration"
            )

    def log_tracking(self, active_count: int, frame_number: int) -> None:
        """Logs periodic tracking state."""
        self.logger.debug(
            f"[TRACKING] Frame #{frame_number} | Active in-frame tracks: {active_count}"
        )

    def flush(self, timeout: float = 2.0):
        """Waits for pending async disk/DB tasks to complete."""
        try:
            self._async_queue.join()
        except Exception:
            pass

    def stop(self):
        """Stops the worker thread cleanly."""
        self.flush()
        self._worker_running = False

    def get_recent_logs(self, limit: int = 30) -> List[str]:
        """Returns the most recent log lines from memory buffer."""
        logs = list(self.recent_logs)
        return logs[-limit:] if len(logs) > limit else logs

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

