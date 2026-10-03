import os

# Cap thread pools for efficient execution without CPU contention
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

try:
    from insightface.utils import face_align
    HAS_FACE_ALIGN = True
except Exception:
    HAS_FACE_ALIGN = False

class FaceDetectionResult:
    """Encapsulates detected face attributes."""
    def __init__(
        self,
        bbox: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        confidence: float,
        landmarks: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        raw_face_obj: Optional[Any] = None
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.landmarks = landmarks
        self.embedding = embedding
        self.raw_face_obj = raw_face_obj

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceDetector:
    """
    High-Performance Multi-Backend Face Detector & ArcFace Embedding Extractor.
    Supports:
    - InsightFace (Buffalo_sc / Buffalo_l) with optimized detection scales (480x480 / 640x640)
    - High-accuracy landmark-aligned ArcFace embedding extraction
    - YOLOv8 Face / Object Detector with integrated ArcFace embedding extraction
    - OpenCV DNN Fallback
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        det_cfg = config.get("detection", {})
        self.framework = det_cfg.get("framework", "insightface").lower()
        self.model_name = det_cfg.get("model_name", "buffalo_sc")
        self.conf_threshold = det_cfg.get("confidence_threshold", 0.25)
        self.min_size = det_cfg.get("min_face_size", [8, 8])
        self.nms_threshold = det_cfg.get("nms_threshold", 0.40)
        
        # Optimized detection input resolution
        det_size_cfg = det_cfg.get("det_size", [480, 480])
        if isinstance(det_size_cfg, list) and len(det_size_cfg) == 2:
            self.det_size = (int(det_size_cfg[0]), int(det_size_cfg[1]))
        else:
            self.det_size = (480, 480)

        self.app = None
        self.det_model = None
        self.rec_model = None
        self.yolo_model = None
        self.backend = "opencv"

        self._init_models()

    def _init_models(self):
        """Initializes detection and recognition engines with GPU/CPU provider detection."""
        providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

        # 1. Initialize InsightFace Backend
        try:
            import insightface
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name=self.model_name,
                providers=providers,
                allowed_modules=['detection', 'recognition']
            )
            self.app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.conf_threshold)
            self.det_model = self.app.models.get('detection')
            self.rec_model = self.app.models.get('recognition')
            self.backend = "insightface"
        except Exception as e:
            print(f"[FaceDetector] InsightFace initialization info/fallback: {e}")
            self.app = None
            self.det_model = None

        # 2. Initialize YOLO Backend if configured or as alternative
        if self.framework == "yolo" or self.backend != "insightface":
            try:
                from ultralytics import YOLO
                yolo_path = "yolov8n.pt"
                if os.path.exists(yolo_path):
                    self.yolo_model = YOLO(yolo_path)
                    if self.framework == "yolo":
                        self.backend = "yolo"
            except Exception as e:
                print(f"[FaceDetector] YOLO initialization info: {e}")

    def detect(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detects faces in the given BGR frame with optimized high-speed inference.
        Returns a list of FaceDetectionResult objects with bounding box and keypoints.
        """
        if frame is None or frame.size == 0:
            return []

        results: List[FaceDetectionResult] = []
        h, w = frame.shape[:2]

        if self.backend == "insightface":
            if self.det_model is not None:
                try:
                    bboxes, kpss = self.det_model.detect(frame, max_num=0, metric='default')
                    for i in range(len(bboxes)):
                        box = bboxes[i]
                        conf = float(box[4]) if len(box) > 4 else 1.0
                        if conf < self.conf_threshold:
                            continue

                        x1 = max(0, min(w - 1, int(box[0])))
                        y1 = max(0, min(h - 1, int(box[1])))
                        x2 = max(x1 + 1, min(w, int(box[2])))
                        y2 = max(y1 + 1, min(h, int(box[3])))

                        bw = x2 - x1
                        bh = y2 - y1
                        if bw < self.min_size[0] or bh < self.min_size[1]:
                            continue

                        kps = kpss[i] if (kpss is not None and i < len(kpss)) else None

                        results.append(
                            FaceDetectionResult(
                                bbox=(x1, y1, x2, y2),
                                confidence=conf,
                                landmarks=kps,
                                embedding=None,
                                raw_face_obj=box
                            )
                        )
                except Exception as e:
                    print(f"[FaceDetector] Fast detection error: {e}")
            elif self.app is not None:
                faces = self.app.get(frame)
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1 = max(0, min(w - 1, int(bbox[0])))
                    y1 = max(0, min(h - 1, int(bbox[1])))
                    x2 = max(x1 + 1, min(w, int(bbox[2])))
                    y2 = max(y1 + 1, min(h, int(bbox[3])))
                    if (x2 - x1) < self.min_size[0] or (y2 - y1) < self.min_size[1]:
                        continue
                    conf = float(face.det_score) if hasattr(face, "det_score") else 1.0
                    if conf < self.conf_threshold:
                        continue
                    results.append(
                        FaceDetectionResult(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            landmarks=getattr(face, 'kps', None),
                            embedding=getattr(face, 'embedding', None),
                            raw_face_obj=face
                        )
                    )

        elif self.backend == "yolo" and self.yolo_model is not None:
            yolo_res = self.yolo_model(frame, verbose=False, conf=self.conf_threshold)
            for r in yolo_res:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1 = max(0, min(w - 1, int(xyxy[0])))
                    y1 = max(0, min(h - 1, int(xyxy[1])))
                    x2 = max(x1 + 1, min(w, int(xyxy[2])))
                    y2 = max(y1 + 1, min(h, int(xyxy[3])))

                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < self.min_size[0] or bh < self.min_size[1]:
                        continue

                    results.append(
                        FaceDetectionResult(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            landmarks=None,
                            embedding=None,
                            raw_face_obj=box
                        )
                    )

        return results

    def extract_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], landmarks: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Extracts a 512-D normalized ArcFace embedding with landmark alignment if available.
        """
        if frame is None or frame.size == 0:
            return None

        # 1. If 5 landmarks are available, use standard InsightFace affine alignment
        if landmarks is not None and HAS_FACE_ALIGN and self.rec_model is not None:
            try:
                aligned = face_align.norm_crop(frame, landmark=landmarks, image_size=112)
                feat = self.rec_model.get_feat(aligned)
                if feat is not None:
                    feat = np.array(feat, dtype=np.float32).flatten()
                    norm = np.linalg.norm(feat)
                    if norm > 1e-6:
                        feat = feat / norm
                    return feat
            except Exception:
                pass

        # 2. Crop directly from frame with padding
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad_w = int((x2 - x1) * 0.15)
        pad_h = int((y2 - y1) * 0.15)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)
        crop = frame[cy1:cy2, cx1:cx2]

        return self.extract_embedding_from_crop(crop)

    def extract_embedding_from_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Directly extracts a 512-D ArcFace embedding from a cropped face image.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        if self.rec_model is not None and hasattr(self.rec_model, 'get_feat'):
            try:
                aligned = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                feat = self.rec_model.get_feat(aligned)
                if feat is not None:
                    feat = np.array(feat, dtype=np.float32).flatten()
                    norm = np.linalg.norm(feat)
                    if norm > 1e-6:
                        feat = feat / norm
                    return feat
            except Exception:
                pass

        if self.app is not None:
            try:
                faces = self.app.get(face_crop)
                if faces and hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
                    emb = faces[0].embedding.flatten()
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        emb = emb / norm
                    return emb
            except Exception:
                pass

        return None
