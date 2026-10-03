import os
import sys
import json
import time
import signal
import argparse
import cv2
from datetime import datetime

from core.pipeline import FaceTrackerPipeline

def load_config(config_path: str = "config.json") -> dict:
    """Loads configuration with fallback defaults."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[Warning] {config_path} not found. Using defaults.")
    return {
        "stream": {"source": "data/record_20250620_183903.mp4", "is_rtsp": False, "target_fps": 25},
        "detection": {"model_name": "buffalo_sc", "confidence_threshold": 0.55, "frame_skip": 3},
        "recognition": {"similarity_threshold": 0.50, "embedding_dim": 512},
        "tracking": {"max_missing_frames": 25, "min_hits_to_confirm": 3},
        "logging": {"log_dir": "logs", "events_log_file": "logs/events.log"},
        "database": {"db_path": "database/visitors.db"}
    }


def main():
    parser = argparse.ArgumentParser(
        description="Katomaran Hackathon: Intelligent Face Tracker with Auto-Registration and Visitor Counting"
    )
    parser.add_argument("--source", type=str, default=None, help="Path to video file or webcam index (e.g. 0)")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP Camera Stream URL (e.g. rtsp://192.168.1.10:554/stream)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--skip", type=int, default=None, help="Number of frames to skip between detection cycles")
    parser.add_argument("--conf", type=float, default=None, help="Face detection confidence threshold")
    parser.add_argument("--sim", type=float, default=None, help="Face recognition cosine similarity threshold")
    parser.add_argument("--display", action="store_true", help="Display live OpenCV window with annotations")
    parser.add_argument("--duration", type=int, default=0, help="Run for specific duration in seconds (0 = infinite/until stream ends)")

    args = parser.parse_args()

    config = load_config(args.config)

    # CLI Overrides
    if args.source is not None:
        config["stream"]["source"] = args.source
        config["stream"]["is_rtsp"] = False
    elif args.rtsp is not None:
        config["stream"]["source"] = args.rtsp
        config["stream"]["is_rtsp"] = True

    if args.skip is not None:
        config["detection"]["frame_skip"] = args.skip
    if args.conf is not None:
        config["detection"]["confidence_threshold"] = args.conf
    if args.sim is not None:
        config["recognition"]["similarity_threshold"] = args.sim

    print("\n" + "="*70)
    print("  KATOMARAN HACKATHON: INTELLIGENT FACE TRACKER & VISITOR COUNTER  ")
    print("="*70)
    print(f" Source Stream        : {config['stream']['source']} (RTSP: {config['stream']['is_rtsp']})")
    print(f" Frame Skip           : {config['detection']['frame_skip']} frames")
    print(f" Detection Model      : {config['detection']['model_name']} (Conf: {config['detection']['confidence_threshold']})")
    print(f" Recognition Engine   : ArcFace 512-D (Similarity Threshold: {config['recognition']['similarity_threshold']})")
    print(f" Mandatory Event Log  : {config['logging']['events_log_file']}")
    print(f" Database Storage     : {config['database']['db_path']}")
    print("="*70 + "\n")

    pipeline = FaceTrackerPipeline(config=config)
    pipeline.start()

    # Graceful shutdown handler
    def shutdown(sig, frame):
        print("\n[Shutdown] Terminating pipeline cleanly...")
        pipeline.stop()
        if args.display:
            cv2.destroyAllWindows()
        print("[Shutdown] Done. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    start_time = time.time()
    try:
        while pipeline.running:
            if args.display:
                frame = pipeline.latest_annotated_frame
                if frame is not None:
                    cv2.imshow("Katomaran Visitor Tracker (Press 'q' to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                time.sleep(0.5)

            # Check if duration limit reached
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\n[Info] Duration limit of {args.duration}s reached.")
                break

            # Print status periodically in console
            summary = pipeline.get_status_summary()
            sys.stdout.write(
                f"\r[Telemetry] Unique Visitors: {summary['unique_visitors']} | "
                f"Active in-frame: {summary['active_in_frame']} | "
                f"Today Entries: {summary['today_entries']} | "
                f"Exits: {summary['today_exits']} | "
                f"FPS: {summary['current_fps']:.1f} | "
                f"CPU: {summary['compute']['cpu_percent']}% | "
                f"RAM: {summary['compute']['memory_mb']} MB"
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if args.display:
            cv2.destroyAllWindows()
        print("\n\n" + "="*70)
        print("  FINAL RUN SUMMARY")
        print("="*70)
        counts = pipeline.db.get_today_event_counts()
        print(f" Total Unique Visitors Counted : {pipeline.db.get_unique_visitor_count()}")
        print(f" Total Entry Events Logged     : {counts['entry']}")
        print(f" Total Exit Events Logged      : {counts['exit']}")
        print(f" Events Log Location           : {os.path.abspath(config['logging']['events_log_file'])}")
        print(f" Database Location             : {os.path.abspath(config['database']['db_path'])}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
