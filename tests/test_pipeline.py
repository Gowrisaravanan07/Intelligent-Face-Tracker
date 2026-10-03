import os
import pytest
import numpy as np
from datetime import datetime
from core.recognizer import FaceRecognizer
from database.db_manager import DatabaseManager
from logger.event_logger import EventLogger

def test_face_reidentification_does_not_increment_unique_count(tmp_path):
    """
    Critical Hackathon Requirement Test:
    Re-identification of the same face in later frames must NOT increment the unique count.
    """
    db_file = tmp_path / "test_visitors.db"
    db = DatabaseManager(db_path=str(db_file), enable_wal=False)
    
    config = {
        "recognition": {
            "similarity_threshold": 0.50,
            "embedding_update_weight": 0.15,
            "embedding_dim": 512
        },
        "logging": {
            "log_dir": str(tmp_path / "logs"),
            "events_log_file": str(tmp_path / "logs" / "events.log"),
            "entries_folder": str(tmp_path / "logs" / "entries"),
            "exits_folder": str(tmp_path / "logs" / "exits")
        }
    }
    
    logger = EventLogger(config=config, db_manager=db)
    recognizer = FaceRecognizer(config=config, db_manager=db, logger=logger)
    
    # 1. Base Embedding for Person A
    base_embedding = np.random.randn(512).astype(np.float32)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    
    # First detection -> Should auto-register VISITOR_0001
    vis_id_1, conf_1, is_new_1 = recognizer.identify_or_register(
        embedding=base_embedding,
        timestamp=datetime.now(),
        quality_score=0.95,
        frame_number=1
    )
    assert is_new_1 is True
    assert vis_id_1 == "VISITOR_0001"
    assert db.get_unique_visitor_count() == 1

    # 2. Later frame detection of same Person A (slight noise, cosine sim > 0.85)
    noise = np.random.randn(512).astype(np.float32) * 0.05
    similar_embedding = base_embedding + noise
    similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

    vis_id_2, conf_2, is_new_2 = recognizer.identify_or_register(
        embedding=similar_embedding,
        timestamp=datetime.now(),
        quality_score=0.92,
        frame_number=50
    )
    assert is_new_2 is False
    assert vis_id_2 == "VISITOR_0001"
    # CRITICAL: Unique visitor count MUST REMAIN 1!
    assert db.get_unique_visitor_count() == 1

    # 3. Detection of Person B (Orthogonal / distinct embedding)
    distinct_embedding = np.random.randn(512).astype(np.float32)
    # Ensure orthogonality to avoid accidental match
    distinct_embedding -= np.dot(distinct_embedding, base_embedding) * base_embedding
    distinct_embedding = distinct_embedding / np.linalg.norm(distinct_embedding)

    vis_id_3, conf_3, is_new_3 = recognizer.identify_or_register(
        embedding=distinct_embedding,
        timestamp=datetime.now(),
        quality_score=0.90,
        frame_number=100
    )
    assert is_new_3 is True
    assert vis_id_3 == "VISITOR_0002"
    # Now unique visitor count increments to 2
    assert db.get_unique_visitor_count() == 2
