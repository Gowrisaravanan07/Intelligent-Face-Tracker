# AI App Planning Document & Feature Specification

## 1. Executive Summary & Problem Formulation
The objective is to engineer a production-grade, real-time computer vision system capable of continuous face detection, multi-object tracking, facial feature embedding generation, automated visitor registration, and unique visitor counting from high-resolution video streams and live RTSP camera feeds.

The system must guarantee:
1. **Idempotent Unique Visitor Counting**: New faces are registered once; re-appearances and re-identifications in later frames must never inflate the unique visitor count.
2. **Deterministic Entry & Exit Logging**: Exactly one `ENTRY` event (with cropped face image, timestamp, bounding box) and exactly one `EXIT` event (with dwell time and exit photo) are logged per physical visit session.
3. **Resilient Dual Storage**: Structured file logging (`events.log`), organized disk image storage (`logs/entries/` and `logs/exits/`), and relational SQLite database persistence.
4. **Real-time Efficiency via Configurable Frame Skipping**: Extrapolating bounding box trajectories using Kalman velocity prediction during skipped frames without dropping track continuity.

---

## 2. Model Selection Rationale
| Module | Selected Technology | Why Chosen | Alternative Avoided | Why Avoided |
| :--- | :--- | :--- | :--- | :--- |
| **Face Detection** | InsightFace RetinaFace / YOLOv8 Face | Superior accuracy under scale, occlusion, and lighting variations; sub-15ms inference. | Haar Cascades / HOG | High false-positive rates; fails on angled or small faces. |
| **Face Recognition** | ArcFace (512-D Normalized Embeddings) | SOTA discriminative margin loss; high angular separation; 99.8% LFW benchmark. | `face_recognition` (dlib 128D) | Explicitly prohibited; poor discriminative capability in crowded retail scenarios. |
| **Tracking** | IoU Kalman Tracker / ByteTrack | Persistent track IDs, zero drift during frame skips, low CPU overhead (< 2ms). | Simple centroid tracker | Fails on crossing trajectories and fast movements. |
| **Database** | SQLite with WAL Mode | Zero external setup, ACID compliant, millisecond vectorized vector queries. | In-memory only | Unsafe against server crashes or power interruptions. |

---

## 3. Comprehensive Feature Document

### Module 1: Video Capture & RTSP Ingestion
- Multi-source support: Pre-recorded video files (`.mp4`, `.avi`, `.mkv`), webcams (`0`), and live RTSP streams (`rtsp://user:pass@ip:port/h264`).
- Threaded decoupled frame buffering to eliminate I/O lag and GUI bottlenecks.
- Automatic reconnect mechanism with configurable retry attempts and backoff intervals.

### Module 2: Detection & Configurable Frame Skipping
- Configurable `frame_skip` parameter loaded from `config.json`.
- Detection executed on frame $k = 0, (frame\_skip + 1), 2 \times (frame\_skip + 1), \dots$.
- Skipped frames utilize Kalman velocity prediction to maintain continuous bounding box coordinates without running expensive neural network forward passes.

### Module 3: Face Recognition & Automated Identity Registration
- ArcFace 512-dimensional L2-normalized feature extraction.
- Vectorized matrix cosine similarity comparison against registered database gallery:
  $$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- **Auto-Registration**: If $\max(\text{Similarity}) < \text{threshold}$, automatically mints a new unique identifier (`VISITOR_0001`), persists embedding, and increments unique visitor count.
- **Re-Identification**: If $\max(\text{Similarity}) \ge \text{threshold}$, associates track with existing identity, logs re-identification, and preserves unique visitor count.
- **Exponential Moving Average (EMA) Embedding Update**: Smoothly incorporates slight facial angle/lighting variations to improve long-term recall.

### Module 4: Idempotent Entry & Exit Logging System
- **Entry Trigger**: Fires when a track reaches $\ge \text{min\_hits\_to\_confirm}$ frames.
  - Crops face with $25\%$ contextual padding.
  - Saves JPEG image to `logs/entries/YYYY-MM-DD/{visitor_id}_{timestamp}_entry.jpg`.
  - Appends structured log line to `logs/events.log`.
  - Inserts record into database `events` table.
- **Exit Trigger**: Fires when track is lost for $> \text{max\_missing\_frames}$ or exits frame boundaries.
  - Saves clearest known exit face crop to `logs/exits/YYYY-MM-DD/{visitor_id}_{timestamp}_exit.jpg`.
  - Computes exact dwell duration: $\Delta t = t_{exit} - t_{entry}$.
  - Appends structured log line to `logs/events.log`.
  - Inserts exit record into database `events` table and updates visitor's total visits.

### Module 5: Real-time Web Dashboard & Telemetry Hub
- Modern glassmorphism dark-mode web application (FastAPI + HTML5/CSS3/Vanilla JS).
- MJPEG live annotated video feed with interactive bounding boxes and HUD stats.
- Real-time KPI cards (Unique Visitors, In Frame Now, Today's Entries, Today's Exits, FPS).
- Registered visitors portrait gallery with session count and timestamps.
- Live Entry & Exit event feed with click-to-enlarge lightbox modal.
- Built-in `events.log` terminal viewer with live auto-scroll.
- Live Canvas chart graphing CPU utilization and processing FPS.
- On-the-fly RTSP switcher and live parameter tuning (`frame_skip`, similarity threshold).
- Instant CSV export of visitor logs.

---

## 4. State Machine & Event Flowchart

```
[ New Frame Ingested ]
        │
        ▼
Is Frame Index % (frame_skip + 1) == 0 ?
   ├── YES ──► Run Face Detection (InsightFace) ──► Hungarian IoU Association
   └── NO  ──► Predict BBoxes via Kalman Motion ──► Update Active Tracks
        │
        ▼
For Each Confirmed Track:
   ├── Has Visitor ID? ──► NO ──► Extract ArcFace 512-D Embedding
   │                                  │
   │                                  ├── Sim >= Thresh ──► Match Existing VISITOR_XXXX (No Count Increment)
   │                                  └── Sim < Thresh  ──► Register New VISITOR_YYYY (Increment Count)
   │                                  │
   │                                  └──► Log EXACTLY ONE [EVENT:ENTRY] (Disk Image + events.log + DB)
   │
   └── Still in Frame? ──► YES ──► Update BBox & Trajectory
        │
        └── NO (Lost > max_missing_frames) ──► Log EXACTLY ONE [EVENT:EXIT] (Dwell Time + Image + DB)
```
