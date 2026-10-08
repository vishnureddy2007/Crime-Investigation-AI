# 🔍 Crime Investigation AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-00FFFF.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/PyTest-25%2F25%20Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Crime Investigation AI** is an advanced, offline-capable forensic analysis and investigation workspace designed for law enforcement, digital forensics units, and criminal investigators. 

The system processes uploaded crime-scene photos and CCTV videos to perform automated object detection, multi-stage weapon verification, timeline reconstruction, situation risk modeling, AI narrative generation, and 2D forensic investigation video rendering.

---

## 🌟 Key Features

### 🎯 1. Multi-Source Object Detection & Tracking
- Powered by **YOLOv8** multi-model pipeline (COCO general model + dedicated `weapon.pt` & `threat_weapon.pt` models).
- Detects persons, vehicles, belongings, and specific weapon classes with high precision.
- Cross-model Non-Maximum Suppression (NMS) and IoU/IoS bounding box deduplication.

### 🛡️ 2. Four-Stage Weapon Verification Engine
- **False Positive Shield**: Eliminates spurious candidate detections caused by noise, background shadows, or non-weapon objects.
- **Subtype Preservation**: Accurately recognizes and verifies specific firearm subtypes including **Revolver**, **Pistol**, **Handgun**, **Rifle**, **Shotgun**, and **Knife**.
- **Multi-Signal Verification**: Evaluates bounding box scale, aspect ratios, source model credibility, and multi-frame temporal persistence for video keyframes.
- **Strict Evidence Isolation**: Distinguishes verified weapons from unverified candidates. Unverified candidates are excluded from weapon counts and severity scoring.

### 🎬 3. Python-Native 2D Investigation Video Generator
- Renders 1080p forensic investigation videos using pure **Python (OpenCV & Pillow)** — **zero external 3D software or Blender dependencies required**.
- **Visual Evidence Correlation**: Embeds 65–80% canvas area actual crime-scene photos and representative video keyframes.
- **Ken Burns Dynamic Zoom**: Smoothly zooms and pans toward verified weapon regions without distortion.
- **Overlay Cards & Timelines**: Displays dark forensic overlay cards, detection color-coded bounding boxes (Verified Cyan, Candidate Yellow, Rejected Red, Person Blue), chronological timeline visualizer, and final investigation summary.

### 🧠 4. AI Crime-Scene Situation Analysis
- Integrates local **Qwen3 14B AI** model (via Ollama) or fallback **Local Evidence Prediction Engine**.
- Generates structured, cautious forensic narratives (*"The visual evidence suggests..."*, *"The available evidence indicates..."*).
- Models likely activity patterns, potential sequences of events, risk indicators, and uncertainty factors without overclaiming.

### 📄 5. Automated Forensic Reporting
- Generates comprehensive **PDF and Word (.docx)** investigative reports.
- Includes complete evidence breakdown tables, weapon verification audit logs, threat metrics, key observations, and investigator signatures.

---

## 🏗️ System Architecture

```
                               ┌────────────────────────┐
                               │  Uploaded Crime Scene  │
                               │   Photos & CCTV Videos │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Multi-Source Detector  │
                               │  YOLOv8 + NMS Dedup    │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ 4-Stage Verification   │
                               │  Scale, Aspect Ratio,  │
                               │  Confidence, Subtype   │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Evidence Analyzer &    │
                               │ Risk Severity Model    │
                               └───────────┬────────────┘
                                           │
             ┌─────────────────────────────┼─────────────────────────────┐
             │                             │                             │
             ▼                             ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│ Investigation Video     │   │ Qwen3 / Local AI        │   │ PDF / DOCX Report       │
│ Rendering Engine        │   │ Narrative Generator     │   │ Export Engine           │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
```

---

## 📁 Repository Structure

```
Crime-Investigation-AI/
├── app.py                         # Streamlit application entry point
├── config/                        # Settings, thresholds, and environment configuration
├── database/                      # SQLite schema DDL and repository interface
├── models/                        # Core AI, detection, and verification modules
│   ├── yolo_detector.py           # MultiSourceDetector & YOLOv8 wrapper
│   ├── weapon_verifier.py         # Multi-stage weapon verification engine
│   ├── evidence_analyzer.py       # Crime severity & threat evaluation
│   ├── summary_generator.py       # AI narrative and template summary builder
│   ├── report_generator.py        # PDF and DOCX forensic report generators
│   └── schemas.py                 # Core Pydantic / Dataclass models
├── services/                      # Background services & video generator
│   └── investigation_video_generator.py # Python-native 2D video rendering engine
├── pages/                         # Streamlit workspace UI pages
│   ├── home.py                    # Case dashboard & quick stats
│   ├── investigation.py           # Primary workspace: upload, detect, analyze
│   ├── investigation_video.py     # Investigation video generation page
│   ├── ai_insights.py             # Situation analysis & Qwen3 narratives
│   ├── report.py                  # Report preview & download center
│   ├── case_history.py            # Historical case repository
│   └── link_analysis.py           # Cross-case evidence correlation graph
├── tests/                         # PyTest test suite (100% passing)
├── outputs/                       # Generated evidence, videos, and reports
└── docs/                          # Developer, installation, and API documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: 3.10 or higher (Python 3.11 / 3.12 recommended)
- **Git**: For cloning repository

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/vishnureddy2007/Crime-Investigation-AI.git
cd Crime-Investigation-AI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Application

```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🔬 Weapon Verification Pipeline & Thresholds

The weapon verification engine relies on a multi-signal decision matrix:

| Threshold Parameter | Value | Description |
| :--- | :--- | :--- |
| `WEAPON_CONF_THRESHOLD` | `0.28` | Minimum raw confidence required for candidate evaluation |
| `WEAPON_VERIFY_THRESHOLD` | `0.50` | Minimum composite score required for verified status |
| `WEAPON_HIGH_CONF_THRESHOLD` | `0.80` | High-confidence override threshold |
| `CLAHE_PASS_CONF_THRESHOLD` | `0.40` | Minimum confidence cutoff during contrast enhancement |

### Verification States:
1. **`VERIFIED_WEAPON`**: High visual evidence confidence. Included in primary weapon counts, threat assessment, and video highlights.
2. **`WEAPON_CANDIDATE`**: Moderate confidence. Marked as unverified candidate requiring manual review; excluded from verified counts.
3. **`NOT_WEAPON`**: Sub-threshold, small artifact box, or extreme aspect ratio detection. Rejected and logged.

---

## 🧪 Testing & Verification

Run the full suite of unit and integration tests:

```bash
# Run all tests
pytest

# Run weapon verification acceptance suite
pytest tests/test_weapon_verification_comprehensive.py

# Run video generator test suite
pytest tests/test_investigation_video_generator.py
```

All **25 core tests** run cleanly in ~23 seconds with **100% pass rate**.

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

## 🤝 Authors & Acknowledgments

- **Developer**: Vishnu Reddy
- **Repository**: [Crime-Investigation-AI on GitHub](https://github.com/vishnureddy2007/Crime-Investigation-AI)