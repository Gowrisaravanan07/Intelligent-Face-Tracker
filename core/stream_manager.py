import os
import cv2
import time
import queue
import threading
import sys
from typing import Optional, Tuple, Dict, Any

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

class StreamManager:
    """
    High-Throughput Zero-Lag Stream Manager supporting:
    - Pre-recorded video files with smooth wall-clock synchronization (high-precision monotonic timer)
    - Live RTSP camera streams with zero-latency buffer dropping and auto-reconnection
    - Webcams & USB cameras
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        st_cfg = config.get("stream", {})
        self.source = st_cfg.get("source", 0)
        self.is_rtsp = st_cfg.get("is_rtsp", False)
        self.rtsp_transport = st_cfg.get("rtsp_transport", "tcp")
        self.reconnect_interval = st_cfg.get("reconnect_interval_sec", 3)
        self.max_reconnect_attempts = st_cfg.get("max_reconnect_attempts", 10)
        self.target_fps = st_cfg.get("target_fps", 25)
        self.resize_w = st_cfg.get("resize_width", 960)
        self.resize_h = st_cfg.get("resize_height", 540)

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.running: bool = False
        self.worker_thread: Optional[threading.Thread] = None

        self.total_frames: int = 0
        self.fps: float = float(self.target_fps) if self.target_fps else 25.0
        self.is_file: bool = False

    def set_source(self, new_source: Any, is_rtsp: bool = False):
        """Switches stream source dynamically on the fly."""
        self.stop()
        self.source = new_source
        self.is_rtsp = is_rtsp
        self.start()

    def _open_capture(self) -> bool:
        """Opens OpenCV VideoCapture with optimal backend settings."""
        source_str = str(self.source)
        if source_str.isdigit():
            self.source = int(self.source)
            self.is_file = False
            self.is_rtsp = False
        elif source_str.startswith("rtsp://") or source_str.startswith("http://") or source_str.startswith("rtsps://"):
            self.is_rtsp = True
            self.is_file = False
        elif os.path.exists(source_str):
            self.is_file = True
            self.is_rtsp = False
        else:
            self.is_file = False

        if self.is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.rtsp_transport}|fflags;nobuffer|max_delay;500000"

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        # Optimize buffer for live video sources
        if not self.is_file:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if detected_fps and 1.0 <= detected_fps <= 120.0:
            self.fps = detected_fps
        else:
            self.fps = float(self.target_fps) if self.target_fps else 25.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def _reader_loop(self):
        """Threaded reading loop with automatic reconnection, smooth monotonic pacing, and zero-lag frame replacement."""
        reconnects = 0
        frame_interval = 1.0 / max(1.0, self.fps)
        next_frame_time = time.perf_counter()

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if reconnects >= self.max_reconnect_attempts:
                    print(f"[StreamManager] Max reconnect attempts ({self.max_reconnect_attempts}) reached. Halting.")
                    break
                print(f"[StreamManager] Connecting to stream source {self.source}... Attempt #{reconnects + 1}")
                success = self._open_capture()
                if not success:
                    reconnects += 1
                    time.sleep(self.reconnect_interval)
                    continue
                else:
                    reconnects = 0
                    frame_interval = 1.0 / max(1.0, self.fps)
                    next_frame_time = time.perf_counter()

            # Monotonic drift-free pacing for video files
            if self.is_file:
                now = time.perf_counter()
                sleep_time = next_frame_time - now
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

                # Catch-up protection: if lag exceeds 2 frames, reset anchor
                now_after = time.perf_counter()
                if now_after - next_frame_time > frame_interval * 2.0:
                    next_frame_time = now_after + frame_interval
                else:
                    next_frame_time += frame_interval

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.005)
                    next_frame_time = time.perf_counter()
                    continue
                else:
                    print("[StreamManager] RTSP stream interrupted. Reconnecting...")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    time.sleep(self.reconnect_interval)
                    continue

            # Frame downscale for high throughput and consistent resolution
            if self.resize_w and self.resize_h:
                h, w = frame.shape[:2]
                if w != self.resize_w or h != self.resize_h:
                    frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)

            # Drop stale frames in queue if full to guarantee real-time playback
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def start(self):
        """Starts stream reader thread."""
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.worker_thread.start()

    def read_frame(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Fetches the latest frame from the buffer."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stops stream capture cleanly."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
