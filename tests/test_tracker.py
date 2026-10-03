import pytest
import numpy as np
from datetime import datetime
from core.tracker import MultiObjectTracker, compute_iou

def test_compute_iou():
    boxA = (10, 10, 50, 50)
    boxB = (10, 10, 50, 50)
    assert compute_iou(boxA, boxB) == 1.0

    boxC = (100, 100, 150, 150)
    assert compute_iou(boxA, boxC) == 0.0

    boxD = (30, 30, 70, 70)
    iou = compute_iou(boxA, boxD)
    assert 0.0 < iou < 1.0

def test_tracker_lifecycle_and_frame_skip():
    config = {
        "tracking": {
            "max_missing_frames": 5,
            "min_hits_to_confirm": 2,
            "iou_threshold": 0.3
        }
    }
    tracker = MultiObjectTracker(config=config)
    now = datetime.now()

    # Frame 1: First detection
    dets_frame1 = [((100, 100, 200, 200), 0.9, np.zeros((100, 100, 3)), np.zeros(512))]
    tracks = tracker.update_with_detections(dets_frame1, timestamp=now, frame_number=1)
    assert len(tracks) == 1
    assert tracks[0].state == "tentative"
    assert tracks[0].hits == 1

    # Frame 2: Second detection (Promotion to Confirmed)
    dets_frame2 = [((102, 102, 202, 202), 0.92, np.zeros((100, 100, 3)), np.zeros(512))]
    tracks = tracker.update_with_detections(dets_frame2, timestamp=now, frame_number=2)
    assert len(tracks) == 1
    assert tracks[0].state == "confirmed"
    assert tracks[0].hits == 2

    # Frame 3-5: Frame skips (prediction without detection)
    tracker.step_without_detection()
    tracker.step_without_detection()
    tracker.step_without_detection()
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].time_since_update == 3

    # Frame 6-8: Missed beyond max_missing -> Marked as lost
    tracker.step_without_detection()
    tracker.step_without_detection()
    tracker.step_without_detection()
    assert tracker.tracks[0].state == "lost"

    # Clean dead tracks
    dead = tracker.remove_dead_tracks()
    assert len(dead) == 1
    assert len(tracker.tracks) == 0
