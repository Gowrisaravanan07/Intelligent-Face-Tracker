import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, Any, List

from database.db_manager import DatabaseManager
from logger.event_logger import EventLogger

class FaceRecognizer:
    """
    State-of-the-Art Face Recognition and Auto-Registration Engine.
    Uses ArcFace 512-D normalized embeddings and Vectorized Cosine Similarity.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        db_manager: DatabaseManager,
        logger: EventLogger
    ):
        self.config = config
        self.db = db_manager
        self.logger = logger

        rec_cfg = config.get("recognition", {})
        self.similarity_threshold = rec_cfg.get("similarity_threshold", 0.50)
        self.ema_alpha = rec_cfg.get("embedding_update_weight", 0.15)
        self.min_quality = rec_cfg.get("min_quality_score", 0.40)
        self.embedding_dim = rec_cfg.get("embedding_dim", 512)

        # In-memory vector cache for high-speed matrix cosine comparison
        self.known_ids: List[str] = []
        self.known_embeddings: np.ndarray = np.empty((0, self.embedding_dim), dtype=np.float32)

        self._load_known_embeddings()

    def _load_known_embeddings(self):
        """Loads all existing registered embeddings from database into RAM."""
        ids, matrix = self.db.get_all_embeddings()
        self.known_ids = ids
        self.known_embeddings = matrix
        self.logger.info(f"[FaceRecognizer] Loaded {len(self.known_ids)} registered visitor embeddings from DB.")

    def reload(self):
        """Reloads embeddings from DB after reset."""
        self._load_known_embeddings()

    def normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalizes an embedding vector."""
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return vec
        return (vec / norm).astype(np.float32)

    def identify_or_register(
        self,
        embedding: np.ndarray,
        timestamp: datetime,
        thumbnail_path: Optional[str] = None,
        quality_score: float = 1.0,
        frame_number: int = 0
    ) -> Tuple[str, float, bool]:
        """
        Matches embedding against registered database.
        - If matched (sim >= threshold): Returns (visitor_id, similarity_score, is_new=False)
        - If not matched: Automatically registers new visitor and returns (visitor_id, 1.0, is_new=True)
        """
        if embedding is None or len(embedding) == 0:
            return "UNKNOWN", 0.0, False

        norm_emb = self.normalize_vector(embedding)

        # 1. Check against known gallery
        if len(self.known_ids) > 0 and len(self.known_embeddings) > 0:
            # Vectorized Cosine Similarity: query (1, 512) @ gallery (N, 512).T -> (N,)
            similarities = np.dot(self.known_embeddings, norm_emb)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= self.similarity_threshold:
                matched_id = self.known_ids[best_idx]
                self.logger.log_recognition(
                    visitor_id=matched_id,
                    similarity_score=best_score,
                    is_reidentified=True,
                    frame_number=frame_number
                )

                # Smoothly update embedding with EMA
                if best_score >= (self.similarity_threshold + 0.10):
                    self.db.update_visitor_embedding(
                        visitor_id=matched_id,
                        new_embedding=norm_emb,
                        alpha=self.ema_alpha
                    )
                    # Update local cache
                    updated_vec = (1.0 - self.ema_alpha) * self.known_embeddings[best_idx] + self.ema_alpha * norm_emb
                    self.known_embeddings[best_idx] = self.normalize_vector(updated_vec)

                return matched_id, best_score, False
            else:
                self.logger.log_recognition(
                    visitor_id="None",
                    similarity_score=best_score,
                    is_reidentified=False,
                    frame_number=frame_number
                )

        # 2. Auto-register new visitor
        new_visitor_number = len(self.known_ids) + 1
        new_visitor_id = f"VISITOR_{new_visitor_number:04d}"

        success = self.db.register_visitor(
            visitor_id=new_visitor_id,
            embedding=norm_emb,
            first_seen=timestamp,
            thumbnail_path=thumbnail_path,
            quality_score=quality_score
        )

        if success:
            self.known_ids.append(new_visitor_id)
            if len(self.known_embeddings) == 0:
                self.known_embeddings = norm_emb.reshape(1, -1)
            else:
                self.known_embeddings = np.vstack([self.known_embeddings, norm_emb.reshape(1, -1)])

            self.logger.log_registration(
                visitor_id=new_visitor_id,
                embedding_dim=len(norm_emb),
                quality_score=quality_score,
                frame_number=frame_number
            )

        return new_visitor_id, 1.0, True
