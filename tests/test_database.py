import os
import pytest
import numpy as np
from datetime import datetime
from database.db_manager import DatabaseManager

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_visitors.db"
    db = DatabaseManager(db_path=str(db_file), enable_wal=False)
    return db

def test_visitor_registration(temp_db):
    emb = np.random.randn(512).astype(np.float32)
    now = datetime.now()
    
    # 1. Register new visitor
    success = temp_db.register_visitor(
        visitor_id="VISITOR_0001",
        embedding=emb,
        first_seen=now,
        quality_score=0.92
    )
    assert success is True
    assert temp_db.get_unique_visitor_count() == 1

    # 2. Duplicate registration attempt should return False
    dup_success = temp_db.register_visitor(
        visitor_id="VISITOR_0001",
        embedding=emb,
        first_seen=now
    )
    assert dup_success is False
    assert temp_db.get_unique_visitor_count() == 1

def test_embedding_retrieval_and_update(temp_db):
    emb1 = np.random.randn(512).astype(np.float32)
    emb2 = np.random.randn(512).astype(np.float32)
    now = datetime.now()

    temp_db.register_visitor("VISITOR_0001", emb1, now)
    temp_db.register_visitor("VISITOR_0002", emb2, now)

    ids, matrix = temp_db.get_all_embeddings()
    assert len(ids) == 2
    assert "VISITOR_0001" in ids
    assert "VISITOR_0002" in ids
    assert matrix.shape == (2, 512)

    # Test EMA update
    new_emb = np.random.randn(512).astype(np.float32)
    updated = temp_db.update_visitor_embedding("VISITOR_0001", new_emb, alpha=0.2)
    assert updated is True

def test_event_logging(temp_db):
    now = datetime.now()
    temp_db.register_visitor("VISITOR_0001", np.random.randn(512).astype(np.float32), now)

    # Log Entry
    entry_id = temp_db.log_event(
        visitor_id="VISITOR_0001",
        event_type="entry",
        timestamp=now,
        image_path="logs/entries/2026-10-02/VISITOR_0001_entry.jpg",
        confidence=0.95,
        frame_number=10
    )
    assert entry_id > 0

    # Log Exit
    exit_id = temp_db.log_event(
        visitor_id="VISITOR_0001",
        event_type="exit",
        timestamp=now,
        image_path="logs/exits/2026-10-02/VISITOR_0001_exit.jpg",
        confidence=1.0,
        session_duration_sec=12.5,
        frame_number=60
    )
    assert exit_id > 0

    counts = temp_db.get_today_event_counts()
    assert counts["entry"] == 1
    assert counts["exit"] == 1

    events = temp_db.get_recent_events(limit=10)
    assert len(events) == 2
