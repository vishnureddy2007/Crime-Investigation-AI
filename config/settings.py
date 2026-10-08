"""
Application configuration settings.

Centralizes paths, model names, and feature flags so other
modules can read consistent values without hardcoding them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ------------------------------------------------------------------
# Deployment-time overrides (Milestone 12+)
# ------------------------------------------------------------------
# All values fall back to safe local defaults so the package remains
# fully usable without any environment configuration.
APP_PORT:        int   = int(os.environ.get("APP_PORT", "8501"))
APP_ADDRESS:     str   = os.environ.get("APP_ADDRESS", "localhost")
APP_HEADLESS:    bool  = os.environ.get("APP_HEADLESS", "false").lower() in {
    "1", "true", "yes", "on",
}
APP_THEME:       str   = os.environ.get("APP_THEME", "light")
_HF_HOME_DEFAULT: Path = Path.home() / ".cache" / "huggingface"
HF_HOME:         Path  = Path(os.environ.get("HF_HOME", str(_HF_HOME_DEFAULT)))
LOG_LEVEL:       str   = os.environ.get("LOG_LEVEL", "INFO").upper()

# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR:       Path = PROJECT_ROOT / "data"
OUTPUTS_DIR:    Path = PROJECT_ROOT / "outputs"
IMAGES_DIR:     Path = PROJECT_ROOT / "images"
MODELS_DIR:     Path = PROJECT_ROOT / "models"
DATABASE_DIR:   Path = PROJECT_ROOT / "database"
DATABASE_PATH:   Path = DATABASE_DIR / "investigations.db"
DOCS_DIR:       Path = PROJECT_ROOT / "docs"
TESTS_DIR:       Path = PROJECT_ROOT / "tests"
LOGS_DIR:        Path = PROJECT_ROOT / "logs"
LOG_FILE:        Path  = Path(os.environ.get("LOG_FILE", str(LOGS_DIR / "app.log")))

REPORTS_DIR:    Path = OUTPUTS_DIR / "reports"
STORYBOARD_DIR: Path = OUTPUTS_DIR / "storyboard"
VIDEOS_DIR:     Path = OUTPUTS_DIR / "videos"
DETECTION_DIR:  Path = OUTPUTS_DIR / "detection"
KEYFRAMES_DIR:  Path = OUTPUTS_DIR / "keyframes"

# ------------------------------------------------------------------
# Model configuration (Milestone 2+)
# ------------------------------------------------------------------
YOLO_MODEL_NAME: str = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD: float = 0.25
YOLO_IOU_THRESHOLD: float = 0.45

# ----------------------------------------------------------------------
# Weapon detection (multi-model setup, two-stage verification)
# ----------------------------------------------------------------------
# Phase 46 — honest two-stage weapon detection.
#
# COCO's `yolov8n.pt` has NO classes for firearms (gun / pistol /
# rifle / handgun). It DOES contain `knife`, which is a real weapon
# class we can claim to detect. Other COCO labels like `baseball bat`,
# `scissors`, `bottle`, and `explosion` are NOT weapons — mapping them
# to "weapon" produced systematic false positives. The mapping below
# is now conservative: only `knife` becomes a weapon candidate.
#
# A dedicated weapon-trained YOLO weights file (if present at
# `models/weapon.pt`) loads as a SECOND model and adds `gun` /
# `pistol` etc. via the same `weapon` label.
WEAPON_MODEL_NAME: str = "models/weapon.pt"
THREAT_WEAPON_MODEL_NAME: str = "models/threat_weapon.pt"

# COCO classes that the general-purpose model can legitimately call a
# weapon. `knife` is the only COCO class that is itself a weapon;
# everything else (bottle, baseball bat, scissors) is its own distinct
# label and is NOT silently re-classified as a weapon.
COCO_WEAPON_CLASSES: set[str] = {"knife"}

# A separate COCO model run with a lower threshold aimed at picking up
# weapons that the default conf may have dropped.
WEAPON_SCAN_CONF_THRESHOLD: float = 0.15

# --- Two-stage weapon verification & thresholds -----------------------
WEAPON_CONF_THRESHOLD: float = 0.28
WEAPON_VERIFY_THRESHOLD: float = 0.48
WEAPON_HIGH_CONF_THRESHOLD: float = 0.78

WEAPON_VERIFICATION_CONFIG: dict[str, Any] = {
    "candidate_confidence": 0.28,
    "verified_threshold": 0.48,
    "high_confidence_override": 0.78,
    "tiny_box_area_pixels": 25.0,
    "tiny_box_rel_area": 0.0003,
    "duplicate_iou_threshold": 0.35,
    "duplicate_ios_threshold": 0.60,
    "multi_scale_enabled": True,
    "crop_padding_fraction": 0.15,
}

# IoU threshold for cross-model NMS dedup. If two detections from
# different models overlap by more than this fraction, the
# higher-confidence one wins.
CROSS_MODEL_IOU_THRESHOLD: float = 0.35

# --- Temporal video verification (Phase 47) ---------------------------
# A candidate must appear in at least MIN_WEAPON_FRAMES distinct
# frames within MAX_FRAME_GAP seconds, with average confidence
# >= MIN_AVERAGE_WEAPON_CONFIDENCE, to be promoted to verified.
MIN_WEAPON_FRAMES: int = 2
MIN_AVERAGE_WEAPON_CONFIDENCE: float = 0.45
MAX_FRAME_GAP_SEC: float = 2.5

# --- Conservative class mapping (Phase 46) ----------------------------
# Things that ARE weapons (verified-weapon label).
# Things that are NOT weapons keep their own label.
YOLO_CLASS_MAPPING: dict[str, str] = {
    # People and belongings
    "person":       "person",
    "backpack":     "bag",
    "handbag":      "bag",
    "suitcase":     "bag",
    # Vehicles
    "car":          "vehicle",
    "motorcycle":   "vehicle",
    "bus":          "vehicle",
    "truck":        "vehicle",
    # COCO classes that COULD be weapons in context (verified only if
    # WEAPON_VERIFY_THRESHOLD is reached; otherwise they remain the
    # neutral class name "knife" / "bottle" / "scissors" / "baseball bat").
    "knife":        "knife",
    "bottle":       "bottle",
    "scissors":     "scissors",
    "baseball bat": "baseball_bat",
    # Dedicated weapon model classes (gun / pistol / rifle / ...).
    # These are only emitted by a *weapon-trained* model, never by
    # COCO `yolov8n.pt`. We accept both casings because some custom
    # weights files use either.
    "gun":          "weapon",
    "Gun":          "weapon",
    "handgun":      "weapon",
    "Handgun":      "weapon",
    "pistol":       "weapon",
    "Pistol":       "weapon",
    "rifle":        "weapon",
    "Rifle":        "weapon",
    "firearm":      "weapon",
    "Firearm":      "weapon",
    "weapon":       "weapon",
    "Weapon":       "weapon",
    "grenade":      "weapon",
    "Grenade":      "weapon",
    # Common firearm synonyms not in the trained class list but still
    # returned by some weapon datasets (Roboflow weapons dataset etc.).
    "revolver":     "weapon",
    "Revolver":     "weapon",
    "shotgun":      "weapon",
    "Shotgun":      "weapon",
    "explosion":    "explosion",
    "Explosion":    "explosion",
}

# Classes we consider "weapon candidates" during the first-stage scan.
# COCO has none beyond `knife`; the dedicated weapon model adds more.
WEAPON_CANDIDATE_LABELS: set[str] = {
    "weapon", "knife", "revolver", "Revolver", "shotgun", "Shotgun",
    "gun", "Gun", "pistol", "Pistol", "rifle", "Rifle", "handgun", "Handgun",
    "firearm", "Firearm", "grenade", "Grenade"
}

# The unified, honest list of "classes we will ever return". The
# COCO model contributes persons, vehicles, bags, and the weapon
# classes listed above. A dedicated weapon model (if loaded) can
# contribute additional weapon classes (gun/pistol/...).
SUPPORTED_DETECTION_CLASSES: set[str] = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "backpack", "handbag", "suitcase",
    "knife", "bottle", "baseball bat", "scissors",
    "gun", "Gun", "pistol", "Pistol", "rifle", "Rifle", "handgun", "Handgun",
    "firearm", "Firearm", "revolver", "Revolver", "shotgun", "Shotgun",
    "weapon", "Weapon", "grenade", "Grenade"
}

RELEVANT_CLASSES: set[str] = set(YOLO_CLASS_MAPPING.keys())

# ------------------------------------------------------------------
# Video processing (Milestone 3+)
# ------------------------------------------------------------------
FRAME_EXTRACTION_INTERVAL_SEC: float = 1.0
MAX_VIDEO_DURATION_SEC: int = 30
MAX_VIDEO_SIZE_MB: int = 100
KEYFRAME_COUNT: int = 5
KEYFRAME_DIR_NAME: str = "keyframes"

# ------------------------------------------------------------------
# Evidence analysis (Milestone 4+, refined in Phase 46)
# ------------------------------------------------------------------
# Weights for the crime-severity score (0-100).
# "verified_weapon" carries the bulk of the penalty. A "candidate"
# weapon (below the verification threshold) adds a small hint instead
# of the full weight — false accusations are worse than missing a
# borderline object.
SEVERITY_WEIGHTS: dict[str, float] = {
    "verified_weapon": 35.0,
    "candidate_weapon": 8.0,
    "person":          8.0,    # per person
    "vehicle":         5.0,    # presence of any vehicle
    "bag":             3.0,    # per bag
    "high_conf":      10.0,    # bonus if any detection conf > 0.85
}

# Suggested crime category rules: (label, required present labels).
# IMPORTANT: order matters. More specific rules must come first.
#
# Phase 46 — these rules no longer let "knife alone" or "person + bag"
# become "armed robbery". A verified weapon AND a person AND a bag is
# still flagged as robbery, but bare evidence does not automatically
# become a violent-crime accusation.
CATEGORY_RULES: list[tuple[str, set[str]]] = [
    ("robbery",             {"verified_weapon", "person", "bag"}),
    ("assault",             {"verified_weapon", "person"}),
    ("theft",               {"person", "bag"}),
    ("vehicle_incident",    {"vehicle"}),
    ("suspicious_activity", {"person"}),
]

# Severity level thresholds (score -> level) and level -> hex color.
# Order matters: highest threshold wins. Shared by the analyzer and
# the dashboard so visuals and logic can never disagree.
SEVERITY_THRESHOLDS: list[tuple[int, str]] = [
    (75, "critical"),
    (50, "high"),
    (25, "moderate"),
    (0,  "low"),
]

SEVERITY_LEVEL_COLORS: dict[str, str] = {
    "critical":  "#b30000",
    "high":      "#e34a33",
    "moderate":  "#fdae61",
    "low":       "#2ca25f",
}


# Validate at import so a typo ("#b3z000") blows up loudly now, instead
# of surfacing as a useless ValueError from a downstream st.markdown call.
_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
for _lvl, _hex in SEVERITY_LEVEL_COLORS.items():
    if not _HEX_RE.match(_hex):
        raise ValueError(
            f"config.SEVERITY_LEVEL_COLORS[{_lvl!r}]={_hex!r} is not a "
            "valid 3- or 6-digit hex color."
        )

# ------------------------------------------------------------------
# AI summary (Milestone 5+)
# ------------------------------------------------------------------
# The primary AI model for summary and chat (runs via Ollama).
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
SUMMARY_MODEL_NAME: str = OLLAMA_MODEL

# Fallback model if Ollama is unavailable.
SUMMARY_MAX_LENGTH: int = 180
SUMMARY_MIN_LENGTH: int = 40
SUMMARY_MAX_NEW_TOKENS: int = 120
SUMMARY_GENERATION_TIMEOUT_SEC: float = 12.0

AI_CONFIG: dict[str, Any] = {
    "model": OLLAMA_MODEL,
    "base_url": OLLAMA_BASE_URL,
    "max_output_tokens": 200,
    "temperature": 0.2,
    "timeout": 12.0,
    "keep_alive": "1h",
    "chat_history_limit": 5,
}

# ------------------------------------------------------------------
# Report generation (Milestone 6+)
# ------------------------------------------------------------------
REPORT_TITLE: str = "AI Crime Investigation Report"
REPORT_AUTHOR: str = "AI Crime Investigation Assistant"
REPORT_REMARKS: str = (
    "This report was generated by an AI assistant for academic and "
    "demonstration purposes only. It is NOT an official record and "
    "must not be used as evidence in any real-world investigation. "
    "All findings should be reviewed and verified by qualified "
    "human investigators."
)

# ------------------------------------------------------------------
# Storyboard / reconstruction (Milestone 7+)
# ------------------------------------------------------------------
STORYBOARD_MIN_SCENES: int = 4
STORYBOARD_MAX_SCENES: int = 6
STORYBOARD_DEFAULT_DURATION_SEC: float = 1.5
STORYBOARD_FADE_SEC: float = 0.5
STORYBOARD_VIDEO_FPS: int = 24
STORYBOARD_VIDEO_CODEC: str = "mp4v"
STORYBOARD_PANEL_SIZE: tuple[int, int] = (640, 360)

# ------------------------------------------------------------------
# App metadata
# ------------------------------------------------------------------
APP_NAME:    str = "AI Crime Investigation Assistant"
APP_VERSION: str = "1.0.0"
