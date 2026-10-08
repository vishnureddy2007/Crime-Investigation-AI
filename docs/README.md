# Crime Investigation AI — Documentation Index

Welcome to the technical documentation suite for **Crime Investigation AI**. This index outlines the architecture, setup guides, API specifications, and progress milestones for the project.

---

## 📚 Core Documentation Index

| Document | Description / Use Case |
|---|---|
| 🚀 **[INSTALLATION.md](INSTALLATION.md)** | Step-by-step setup guide, virtual environment setup, dependencies, and environment verification. |
| 🐳 **[DEPLOYMENT.md](DEPLOYMENT.md)** | Single-command launcher, Docker container deployment, environment variables, and headless configuration. |
| 🛠️ **[DEVELOPER.md](DEVELOPER.md)** | System architecture, pipeline design, weapon verification logic, video rendering engine, and extension cookbook. |
| 📖 **[API.md](API.md)** | Comprehensive API documentation for models, schemas, dataclass structures, session state keys, and functions. |
| 📊 **[PROGRESS.md](PROGRESS.md)** | Milestone history, feature evolution, test suite growth, and benchmark results. |
| 🎬 **[DEMO_SCENARIOS.md](DEMO_SCENARIOS.md)** | Guided walkthrough scenarios for viva demonstrations, video generator presets, and weapon verification tests. |

---

## 🔍 Architecture & Subsystem Highlights

### 1. Multi-Stage Weapon Verification Pipeline
The system incorporates an honest 4-stage verification gate that sits between raw YOLOv8 object detections and evidence analysis:
- **Subtype Normalization**: Detects and preserves specific firearm classes including **Revolver**, **Pistol**, **Handgun**, **Rifle**, **Shotgun**, **Knife**, and **Firearm**.
- **Scale & Shape Checks**: Rejects tiny artifact bounding boxes (`area < 25px`), extreme aspect ratios (`ratio > 8.0`), or screen-filling boxes (`rel_area > 50%`).
- **Composite Verification Scoring**: Combines detection confidence, dedicated model credibility, subtype specificity, and shape quality to classify objects into `VERIFIED_WEAPON`, `WEAPON_CANDIDATE`, or `NOT_WEAPON`.
- **Candidate Deduplication**: Merges overlapping boxes ($\text{IoU} > 0.35$ or $\text{IoS} > 0.60$) into a single verified detection.

### 2. Python-Native 2D Investigation Video Engine
- Renders 1080p forensic investigation videos using pure **Python (OpenCV & Pillow)** without requiring Blender or external 3D software.
- Embeds actual uploaded crime-scene photos and representative video keyframes (occupying 67% of visual canvas area).
- Features dynamic Ken Burns zoom targeting verified weapon regions, color-coded detection overlays, chronological timeline progression, and AI narrative cards.

---

## 🧪 Verification & Test Suite

All 25 unit and integration tests are verified and passing:

```bash
pytest
```

- `tests/test_weapon_verification_comprehensive.py`: 10/10 Passed
- `tests/test_auto_weapon_verification.py`: 8/8 Passed
- `tests/test_revolver_weapon_category.py`: 3/3 Passed
- `tests/test_investigation_video_generator.py`: 3/3 Passed
- `tests/test_render_pipeline.py`: 1/1 Passed