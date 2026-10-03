# System Architecture & Technical Specifications

## 1. High-Level Modular Architecture

The system is designed with a decoupled, thread-safe, multi-tiered architecture ensuring real-time responsiveness, modularity, and crash resilience.

```mermaid
graph TD
    subgraph Ingestion Layer
        A1[Video File: data/*.mp4] --> B[Stream Manager]
        A2[Live RTSP Camera Stream] --> B
        A3[USB / Web Camera] --> B
    end

    subgraph Core Vision Pipeline
        B -->|Threaded Queue| C[Master Pipeline Controller]
        C --> D{Frame Skip Logic}
        D -->|Every Nth Frame| E[Face Detector: InsightFace]
        D -->|Skipped Frames| F[Kalman Motion Extrapolator]
        E --> G[Multi-Object Tracker: IoU Association]
        F --> G
        G --> H[Face Recognizer: ArcFace 512-D]
        H --> I[Vector Cosine Matcher & EMA Engine]
    end

    subgraph State & Idempotency Layer
        I --> J{Identity Match?}
        J -->|Sim >= Threshold| K[Re-identify Visitor<br/>Count Unchanged]
        J -->|Sim < Threshold| L[Auto-Register New Visitor<br/>Increment Unique Count]
        K --> M[State Machine: Entry/Exit Manager]
        L --> M
    end

    subgraph Dual Persistence Layer
        M -->|Save Cropped Face| N[Filesystem: logs/entries/ & logs/exits/]
        M -->|Append Audit Trail| O[Log File: logs/events.log]
        M -->|Transactional ACID| P[SQLite Database: visitors.db]
    end

    subgraph Presentation & Control Layer
        C --> Q[FastAPI Web Server]
        P --> Q
        Q -->|MJPEG Live Stream| R[Web Dashboard Frontend]
        Q -->|REST API / Telemetry| R
        Q -->|CSV Export| S[Analytics Export]
    end
```

---

## 2. Sequence Diagram: Face Entry & Exit Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant Stream as Stream Manager
    participant Pipe as Pipeline Controller
    participant Det as Face Detector
    participant Track as Multi-Object Tracker
    participant Rec as ArcFace Recognizer
    participant Log as Event Logger
    participant DB as SQLite Database

    Note over Stream,Pipe: Video Capture Cycle
    Stream->>Pipe: Frame Buffer (BGR 1280x720)
    
    alt Is Detection Cycle (Frame % (Skip+1) == 0)
        Pipe->>Det: detect(frame)
        Det-->>Pipe: [BBoxes, Landmarks, Confidence]
        Pipe->>Track: update_with_detections(detections)
    else Skipped Frame
        Pipe->>Track: step_without_detection() (Kalman Prediction)
    end
    
    Track-->>Pipe: Active Tracks List

    opt Track hits >= min_hits & visitor_id is None
        Pipe->>Rec: identify_or_register(embedding)
        Rec->>DB: get_all_embeddings()
        Rec-->>Pipe: (visitor_id, confidence, is_new)
        
        Note over Pipe,Log: Exactly 1 Entry Event Triggered
        Pipe->>Log: log_entry(visitor_id, frame, bbox)
        Log->>Log: Save crop to logs/entries/YYYY-MM-DD/
        Log->>Log: Append [EVENT:ENTRY] to events.log
        Log->>DB: INSERT INTO events (type='entry')
    end

    opt Track missing > max_missing_frames (Visitor Exited)
        Note over Pipe,Log: Exactly 1 Exit Event Triggered
        Pipe->>Log: log_exit(visitor_id, best_crop, dwell_time)
        Log->>Log: Save crop to logs/exits/YYYY-MM-DD/
        Log->>Log: Append [EVENT:EXIT] to events.log
        Log->>DB: INSERT INTO events (type='exit')
        Pipe->>Track: remove_dead_tracks()
    end
```

---

## 3. Database Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    VISITORS ||--o{ EVENTS : "generates"
    SYSTEM_METRICS {
        int id PK
        timestamp timestamp
        float fps
        float cpu_percent
        float gpu_percent
        float memory_mb
        int active_tracks
        int unique_visitors_count
        float frame_process_time_ms
    }

    VISITORS {
        text visitor_id PK "e.g. VISITOR_0001"
        text name "Display Name"
        timestamp first_seen "Initial Entry"
        timestamp last_seen "Latest Activity"
        int total_visits "Count of distinct sessions"
        blob embedding_blob "512-D Float32 Vector"
        int embedding_dim "512"
        text thumbnail_path "Path to portrait"
        float quality_score "Detection confidence"
        timestamp created_at
    }

    EVENTS {
        int event_id PK
        text visitor_id FK
        text event_type "entry | exit"
        timestamp timestamp "Event Time"
        text image_path "Cropped Face Path"
        float confidence "Recognition Score"
        float session_duration_sec "Dwell Time (s)"
        int frame_number "Video Frame"
        timestamp created_at
    }
```
