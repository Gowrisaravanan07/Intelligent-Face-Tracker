import time
import psutil
import threading
from datetime import datetime
from typing import Dict, Any, Optional

class ComputeMonitor:
    """
    Real-time System Resource Consumption Profiler & Latency Benchmark.
    Measures CPU, GPU, Memory, and Pipeline Processing Latencies.
    """
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = False
        self._init_gpu()

        self.start_time = time.time()
        self.frame_count = 0
        self.fps_history = []
        self.latency_history = []

        self.last_det_time_ms = 0.0
        self.last_rec_time_ms = 0.0
        self.last_trk_time_ms = 0.0
        self.last_total_time_ms = 0.0

    def _init_gpu(self):
        """Checks for NVIDIA GPU support via pynvml or PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_name = torch.cuda.get_device_name(0)
                return
        except Exception:
            pass

        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        except Exception:
            self.gpu_available = False
            self.gpu_name = "None (Running on CPU)"

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Returns GPU utilization and VRAM usage."""
        if not self.gpu_available:
            return {"gpu_percent": 0.0, "vram_used_mb": 0.0, "vram_total_mb": 0.0, "gpu_name": self.gpu_name}

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
                return {
                    "gpu_percent": 0.0,  # PyTorch doesn't expose compute load directly
                    "vram_used_mb": round(allocated, 1),
                    "vram_total_mb": round(reserved, 1),
                    "gpu_name": self.gpu_name
                }
        except Exception:
            pass

        return {"gpu_percent": 0.0, "vram_used_mb": 0.0, "vram_total_mb": 0.0, "gpu_name": self.gpu_name}

    def record_frame_latency(self, total_ms: float, det_ms: float = 0.0, rec_ms: float = 0.0, trk_ms: float = 0.0):
        """Records cycle execution times."""
        self.frame_count += 1
        self.last_total_time_ms = total_ms
        self.last_det_time_ms = det_ms
        self.last_rec_time_ms = rec_ms
        self.last_trk_time_ms = trk_ms

        self.latency_history.append(total_ms)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)

    def get_metrics_snapshot(self, current_fps: float = 0.0, active_tracks: int = 0, unique_count: int = 0) -> Dict[str, Any]:
        """Generates a complete telemetry dictionary."""
        cpu_pct = psutil.cpu_percent(interval=None)
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)

        gpu_data = self.get_gpu_metrics()

        avg_latency = (
            sum(self.latency_history) / len(self.latency_history)
            if self.latency_history else self.last_total_time_ms
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "fps": round(current_fps, 1),
            "cpu_percent": round(cpu_pct, 1),
            "gpu_percent": gpu_data.get("gpu_percent", 0.0),
            "gpu_name": self.gpu_name,
            "memory_mb": round(mem_mb, 1),
            "active_tracks": active_tracks,
            "unique_visitors_count": unique_count,
            "frame_process_time_ms": round(avg_latency, 2),
            "detection_time_ms": round(self.last_det_time_ms, 2),
            "recognition_time_ms": round(self.last_rec_time_ms, 2),
            "tracking_time_ms": round(self.last_trk_time_ms, 2)
        }
