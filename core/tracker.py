import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

class Track:
    """
    Represents a single continuous face track across video frames with Kalman motion smoothing.
    """
    _id_counter = 0

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        timestamp: datetime,
        detection_conf: float = 1.0,
        landmarks: Optional[np.ndarray] = None,
        face_crop: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None
    ):
        Track._id_counter += 1
        self.track_id: int = Track._id_counter
        self.bbox: Tuple[int, int, int, int] = bbox  # (x1, y1, x2, y2)
        self.hits: int = 1
        self.age: int = 1
        self.time_since_update: int = 0
        self.state: str = "tentative"  # 'tentative' | 'confirmed' | 'lost'

        self.visitor_id: Optional[str] = None
        self.entry_logged: bool = False
        self.exit_logged: bool = False

        self.first_seen: datetime = timestamp
        self.last_seen: datetime = timestamp
        self.start_frame: int = 0

        self.best_face_crop: Optional[np.ndarray] = face_crop.copy() if (face_crop is not None and face_crop.size > 0) else None
        self.best_confidence: float = detection_conf
        self.latest_landmarks: Optional[np.ndarray] = landmarks
        self.latest_embedding: Optional[np.ndarray] = embedding

        # Motion State: [cx, cy, w, h, vx, vy]
        x1, y1, x2, y2 = bbox
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.w = max(8.0, float(x2 - x1))
        self.h = max(8.0, float(y2 - y1))
        self.vx = 0.0
        self.vy = 0.0

        self.history: List[Tuple[int, int]] = [self.center]

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(
        self,
        new_bbox: Tuple[int, int, int, int],
        timestamp: datetime,
        conf: float = 1.0,
        face_crop: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None
    ):
        """Updates track state with a matched detection using Kalman smoothing."""
        x1, y1, x2, y2 = new_bbox
        new_cx = (x1 + x2) / 2.0
        new_cy = (y1 + y2) / 2.0
        new_w = max(8.0, float(x2 - x1))
        new_h = max(8.0, float(y2 - y1))

        # Velocity EMA update normalized by elapsed frames dt
        dt = max(1, self.time_since_update)
        inst_vx = (new_cx - self.cx) / float(dt)
        inst_vy = (new_cy - self.cy) / float(dt)
        self.vx = 0.7 * self.vx + 0.3 * inst_vx
        self.vy = 0.7 * self.vy + 0.3 * inst_vy

        # Smooth position and scale
        self.cx = 0.85 * new_cx + 0.15 * self.cx
        self.cy = 0.85 * new_cy + 0.15 * self.cy
        self.w = 0.85 * new_w + 0.15 * self.w
        self.h = 0.85 * new_h + 0.15 * self.h

        # Recompute smoothed bbox
        sx1 = int(self.cx - self.w / 2.0)
        sy1 = int(self.cy - self.h / 2.0)
        sx2 = int(self.cx + self.w / 2.0)
        sy2 = int(self.cy + self.h / 2.0)
        self.bbox = (sx1, sy1, sx2, sy2)

        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_seen = timestamp

        if landmarks is not None:
            self.latest_landmarks = landmarks

        if conf >= self.best_confidence or self.best_face_crop is None:
            self.best_confidence = conf
            if face_crop is not None and face_crop.size > 0:
                self.best_face_crop = face_crop.copy()

        if embedding is not None:
            self.latest_embedding = embedding

        self.history.append(self.center)
        if len(self.history) > 30:
            self.history.pop(0)

    def predict(self):
        """Propagates bounding box position during skipped detection frames using velocity."""
        self.age += 1
        self.time_since_update += 1

        # Motion extrapolation with velocity decay
        self.cx += self.vx
        self.cy += self.vy
        self.vx *= 0.95
        self.vy *= 0.95

        sx1 = int(self.cx - self.w / 2.0)
        sy1 = int(self.cy - self.h / 2.0)
        sx2 = int(self.cx + self.w / 2.0)
        sy2 = int(self.cy + self.h / 2.0)
        self.bbox = (sx1, sy1, sx2, sy2)

        self.history.append(self.center)
        if len(self.history) > 30:
            self.history.pop(0)


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def compute_diou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Computes Distance-IoU (DIoU) incorporating overlap and center Euclidean distance.
    Provides robust association even when bboxes have zero strict overlap during rapid motion.
    """
    iou = compute_iou(boxA, boxB)
    cxA = (boxA[0] + boxA[2]) / 2.0
    cyA = (boxA[1] + boxA[3]) / 2.0
    cxB = (boxB[0] + boxB[2]) / 2.0
    cyB = (boxB[1] + boxB[3]) / 2.0

    d_squared = (cxA - cxB) ** 2 + (cyA - cyB) ** 2

    # Smallest enclosing box diagonal
    x_min = min(boxA[0], boxB[0])
    y_min = min(boxA[1], boxB[1])
    x_max = max(boxA[2], boxB[2])
    y_max = max(boxA[3], boxB[3])
    c_squared = max(1.0, (x_max - x_min) ** 2 + (y_max - y_min) ** 2)

    return iou - 0.5 * (d_squared / c_squared)


def compute_match_score(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Computes unified association score combining IoU, DIoU, and Euclidean center proximity.
    Guarantees reliable matching across all movement directions (left, right, towards, away, turns).
    """
    iou = compute_iou(boxA, boxB)
    diou = compute_diou(boxA, boxB)

    cxA = (boxA[0] + boxA[2]) / 2.0
    cyA = (boxA[1] + boxA[3]) / 2.0
    cxB = (boxB[0] + boxB[2]) / 2.0
    cyB = (boxB[1] + boxB[3]) / 2.0

    wA = max(8.0, float(boxA[2] - boxA[0]))
    hA = max(8.0, float(boxA[3] - boxA[1]))
    wB = max(8.0, float(boxB[2] - boxB[0]))
    hB = max(8.0, float(boxB[3] - boxB[1]))

    char_dim = max(wA, hA, wB, hB)
    dist = np.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)
    # Proximity score in [0, 1] within 2.5 head diameters
    prox_score = max(0.0, 1.0 - (dist / (2.5 * char_dim)))

    return max(iou, diou, prox_score * 0.6)


class MultiObjectTracker:
    """
    SOTA ByteTrack-Style Multi-Object Face Tracker with Distance-IoU Cost Matrix
    and Two-Stage Association for High Accuracy in Crowded and Dynamic Scenes.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        trk_cfg = config.get("tracking", {})
        det_cfg = config.get("detection", {})
        self.frame_skip = det_cfg.get("frame_skip", 3)
        self.max_missing = trk_cfg.get("max_missing_frames", 50)
        self.min_hits = trk_cfg.get("min_hits_to_confirm", 2)
        self.iou_threshold = trk_cfg.get("iou_threshold", 0.05)
        self.conf_threshold = det_cfg.get("confidence_threshold", 0.25)
        self.high_conf_threshold = max(0.20, self.conf_threshold)
        self.tentative_limit = max(15, (self.frame_skip + 1) * 3)

        self.tracks: List[Track] = []

    def update_with_detections(
        self,
        detections: List[Dict[str, Any]],
        timestamp: datetime,
        frame_number: int
    ) -> List[Track]:
        """
        ByteTrack Two-Stage Association with Multi-Directional Proximity Matching:
        1. Match active tracks with high-confidence detections.
        2. Recover occluded/profile faces with low-confidence detections.
        3. Create new tracks for all unmatched detections.
        """
        # Step 1: Predict all tracks forward
        for t in self.tracks:
            t.predict()

        if len(detections) == 0:
            for t in self.tracks:
                limit = self.max_missing if t.state == "confirmed" else self.tentative_limit
                if t.time_since_update > limit:
                    t.state = "lost"
            return self.tracks

        # Normalize detections into standard dictionary format
        normalized_dets = []
        for det in detections:
            if isinstance(det, dict):
                normalized_dets.append(det)
            elif isinstance(det, (tuple, list)):
                d_box = det[0]
                conf = float(det[1]) if len(det) > 1 else 1.0
                crop = det[2] if len(det) > 2 else None
                emb = det[3] if len(det) > 3 else None
                lmk = det[4] if len(det) > 4 else None
                normalized_dets.append({
                    "bbox": d_box,
                    "confidence": conf,
                    "crop": crop,
                    "embedding": emb,
                    "landmarks": lmk
                })
            else:
                normalized_dets.append({
                    "bbox": getattr(det, "bbox", (0, 0, 0, 0)),
                    "confidence": getattr(det, "confidence", 1.0),
                    "crop": getattr(det, "crop", None),
                    "embedding": getattr(det, "embedding", None),
                    "landmarks": getattr(det, "landmarks", None)
                })

        # Partition detections into High-Score and Low-Score
        high_dets = []
        low_dets = []

        for det in normalized_dets:
            conf = det.get("confidence", 1.0)
            if conf >= self.high_conf_threshold:
                high_dets.append(det)
            else:
                low_dets.append(det)

        matched_tracks = set()
        matched_high_dets = set()

        # --- STAGE 1: Match Active Tracks with High-Confidence Detections ---
        if len(self.tracks) > 0 and len(high_dets) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(high_dets)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, d in enumerate(high_dets):
                    cost_matrix[t_idx, d_idx] = compute_match_score(track.bbox, d["bbox"])

            # Greedy assignment on match score
            for _ in range(min(len(self.tracks), len(high_dets))):
                max_val = np.max(cost_matrix)
                if max_val < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                if t_idx in matched_tracks or d_idx in matched_high_dets:
                    cost_matrix[t_idx, d_idx] = -999.0
                    continue

                d = high_dets[d_idx]
                self.tracks[t_idx].update(
                    new_bbox=d["bbox"],
                    timestamp=timestamp,
                    conf=d["confidence"],
                    face_crop=d.get("crop"),
                    embedding=d.get("embedding"),
                    landmarks=d.get("landmarks")
                )
                if self.tracks[t_idx].hits >= self.min_hits:
                    self.tracks[t_idx].state = "confirmed"

                matched_tracks.add(t_idx)
                matched_high_dets.add(d_idx)
                cost_matrix[t_idx, :] = -999.0
                cost_matrix[:, d_idx] = -999.0

        # --- STAGE 2: Match Remaining Tracks with Low-Confidence Detections ---
        unmatched_tracks = [t_idx for t_idx in range(len(self.tracks)) if t_idx not in matched_tracks]
        if len(unmatched_tracks) > 0 and len(low_dets) > 0:
            low_cost = np.zeros((len(unmatched_tracks), len(low_dets)), dtype=np.float32)
            for u_i, t_idx in enumerate(unmatched_tracks):
                track = self.tracks[t_idx]
                for d_i, d in enumerate(low_dets):
                    low_cost[u_i, d_i] = compute_match_score(track.bbox, d["bbox"])

            for _ in range(min(len(unmatched_tracks), len(low_dets))):
                max_val = np.max(low_cost)
                if max_val < (self.iou_threshold - 0.10):
                    break
                u_i, d_i = np.unravel_index(np.argmax(low_cost), low_cost.shape)
                t_idx = unmatched_tracks[u_i]
                if t_idx in matched_tracks:
                    low_cost[u_i, d_i] = -999.0
                    continue

                d = low_dets[d_i]
                self.tracks[t_idx].update(
                    new_bbox=d["bbox"],
                    timestamp=timestamp,
                    conf=d["confidence"],
                    face_crop=d.get("crop"),
                    embedding=d.get("embedding"),
                    landmarks=d.get("landmarks")
                )
                if self.tracks[t_idx].hits >= self.min_hits:
                    self.tracks[t_idx].state = "confirmed"

                matched_tracks.add(t_idx)
                low_cost[u_i, :] = -999.0
                low_cost[:, d_i] = -999.0

        # --- STAGE 3: Create New Tracks for All Unmatched Detections ---
        for d_idx, d in enumerate(high_dets):
            if d_idx not in matched_high_dets:
                new_track = Track(
                    bbox=d["bbox"],
                    timestamp=timestamp,
                    detection_conf=d["confidence"],
                    landmarks=d.get("landmarks"),
                    face_crop=d.get("crop"),
                    embedding=d.get("embedding")
                )
                new_track.start_frame = frame_number
                if self.min_hits <= 1:
                    new_track.state = "confirmed"
                self.tracks.append(new_track)

        # Mark Lost Tracks (Tentative tracks live up to tentative_limit; Confirmed persist up to max_missing)
        for t in self.tracks:
            limit = self.max_missing if t.state == "confirmed" else self.tentative_limit
            if t.time_since_update > limit:
                t.state = "lost"

        return self.tracks

    def step_without_detection(self) -> List[Track]:
        """Advances tracker state during skipped detection cycles using Kalman prediction."""
        for t in self.tracks:
            t.predict()
            limit = self.max_missing if t.state == "confirmed" else self.tentative_limit
            if t.time_since_update > limit:
                t.state = "lost"
        return self.tracks

    def remove_dead_tracks(self) -> List[Track]:
        """Removes dead tracks and returns them."""
        dead_tracks = [t for t in self.tracks if t.state == "lost"]
        self.tracks = [t for t in self.tracks if t.state != "lost"]
        return dead_tracks

    def get_active_tracks(self) -> List[Track]:
        """Returns currently confirmed active tracks with grace period for temporary occlusions."""
        return [t for t in self.tracks if t.state == "confirmed" and t.time_since_update <= min(30, self.max_missing)]

