"""
YOLOv8 detector wrapper.

Two-tier design:

1. `YOLODetector` — single-model wrapper around Ultralytics YOLO.
   Returns a `DetectionResult` filtered against `RELEVANT_CLASSES` and
   re-mapped via `YOLO_CLASS_MAPPING`.

2. `MultiSourceDetector` — runs one or more `YOLODetector` instances
   and merges their detections with a per-class NMS. The general
   model contributes persons, vehicles, bags, and the small set of
   COCO weapon classes (knife, baseball bat, scissors, bottle). An
   optional weapon-trained model (if its weights file is present)
   contributes additional weapon classes (gun, pistol, ...). The
   merger is honest: a detection is only logged if the model
   actually produced it.

This module is duck-typed for tests. The detector signature does not
depend on Streamlit.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from ultralytics import YOLO

from config import (
    COCO_WEAPON_CLASSES,
    CROSS_MODEL_IOU_THRESHOLD,
    RELEVANT_CLASSES,
    WEAPON_CONF_THRESHOLD,
    WEAPON_MODEL_NAME,
    WEAPON_SCAN_CONF_THRESHOLD,
    YOLO_CLASS_MAPPING,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_MODEL_NAME,
)
from core.observability import RequestTimer
from core.resilience import retry
from models.schemas import BoundingBox, Detection, DetectionResult
from models.weapon_verifier import apply_to_detections as _verify_detections
from utils.visualization import draw_detections


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _ios(a: BoundingBox, b: BoundingBox) -> float:
    """Compute Intersection over Smaller Box Area."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    min_area = min(a.area, b.area)
    return inter / min_area if min_area > 0 else 0.0


def _enhance_contrast(image: Image.Image) -> Image.Image:
    """Enhance metallic edges and dark objects for low-contrast / B&W crime scene photos."""
    try:
        import cv2
        import numpy as np
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_bgr)
    except (RuntimeError, OSError, ValueError, TypeError):
        return image


_WEAPON_LABEL_SYNONYMS: frozenset[str] = frozenset({
    "weapon", "knife", "gun", "pistol", "handgun", "firearm", "rifle",
    "revolver", "Revolver", "shotgun", "Shotgun", "grenade", "Grenade",
    "candidate_weapon", "threat-weapon", "weapon-scan", "Gun", "Pistol", "Handgun"
})


def _cross_model_nms(
    detections: list[Detection],
    iou_threshold: float = CROSS_MODEL_IOU_THRESHOLD,
) -> list[Detection]:
    """
    Suppress overlapping detections across models with multi-stage weapon duplicate merging.

    Iterates by descending confidence, giving priority to dedicated
    weapon-model detections over generic COCO object detections when
    bounding boxes overlap.
    """
    if not detections:
        return []

    def _sort_key(d: Detection) -> float:
        boost = 0.30 if d.source in {"weapon", "threat-weapon"} else 0.0
        return d.confidence + boost

    ordered = sorted(detections, key=_sort_key, reverse=True)
    kept: list[Detection] = []
    for det in ordered:
        duplicate = False
        det_is_weapon = (det.label in _WEAPON_LABEL_SYNONYMS or det.class_name in _WEAPON_LABEL_SYNONYMS)
        
        for kept_det in kept:
            overlap_iou = _iou(det.bbox, kept_det.bbox)
            overlap_ios = _ios(det.bbox, kept_det.bbox)
            kept_is_weapon = (kept_det.label in _WEAPON_LABEL_SYNONYMS or kept_det.class_name in _WEAPON_LABEL_SYNONYMS)

            # 1. Exact same label overlap check
            if det.label == kept_det.label and overlap_iou > iou_threshold:
                duplicate = True
                break

            # 2. Robust Duplicate Weapon Merging across models/synonyms (IoU > 0.35 or IoS > 0.60)
            if det_is_weapon and kept_is_weapon:
                if overlap_iou > 0.35 or overlap_ios > 0.60:
                    duplicate = True
                    break

            # 3. Dedicated weapon model vs COCO non-person/non-vehicle overlap
            if overlap_iou > iou_threshold:
                if (
                    kept_det.source in {"weapon", "threat-weapon"}
                    and det.label not in {"person", "vehicle"}
                ):
                    duplicate = True
                    break
                if (
                    det.source in {"weapon", "threat-weapon"}
                    and kept_det.label not in {"person", "vehicle"}
                ):
                    duplicate = True
                    break

        if not duplicate:
            kept.append(det)
    return kept


# ----------------------------------------------------------------------
# Single-model detector
# ----------------------------------------------------------------------
class YOLODetector:
    """Thin wrapper around Ultralytics YOLO with crime-scene filtering."""

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        confidence: float = YOLO_CONFIDENCE_THRESHOLD,
        iou: float = YOLO_IOU_THRESHOLD,
        *,
        retry_attempts: int = 2,
        retry_initial_delay: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.retry_attempts = retry_attempts
        self.retry_initial_delay = retry_initial_delay
        self._model: YOLO | None = None
        self.last_load_ms: float = 0.0

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    @property
    def model(self) -> YOLO:
        if self._model is None:
            from core.performance import Stopwatch

            with Stopwatch("yolo.load") as sw:
                self._model = YOLO(self.model_name)
            self.last_load_ms = sw.elapsed_ms
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_image(
        self,
        image: Image.Image,
        source_name: str = "image",
        confidence: float | None = None,
        iou: float | None = None,
        allowed_classes: Iterable[str] | None = None,
        source_tag: str = "general",
    ) -> DetectionResult:
        """
        Run detection on a PIL image.

        Parameters
        ----------
        image:           RGB PIL.Image.
        source_name:     Identifier used in the result (filename, etc.).
        confidence:      Optional override for confidence threshold.
        iou:             Optional override for IoU threshold.
        allowed_classes: Subset of class names to keep (None = use
                         RELEVANT_CLASSES). Used by the multi-source
                         detector to give each model its own filter.
        source_tag:      Short label attached to every detection so
                         downstream code can tell which model fired
                         ("general" vs "weapon").
        """
        conf = self.confidence if confidence is None else confidence
        iou_thr = self.iou if iou is None else iou
        keep = set(allowed_classes) if allowed_classes is not None else RELEVANT_CLASSES

        np_img: Any = image

        def _predict() -> object:
            return self.model.predict(
                source=np_img,
                conf=conf,
                iou=iou_thr,
                verbose=False,
            )

        try:
            with RequestTimer("yolo.predict", detail=source_name) as ctx:
                results = retry(
                    _predict,
                    attempts=self.retry_attempts,
                    initial_delay=self.retry_initial_delay,
                    exceptions=(RuntimeError, OSError),
                )
                _ = ctx
        except (RuntimeError, OSError):
            raise

        detections: list[Detection] = []
        raw_count = 0

        if not results:
            return self._build_result(image, source_name, detections, raw_count, source_tag=source_tag)

        result = results[0]
        names: dict[int, str] = result.names
        raw_count = len(result.boxes) if result.boxes is not None else 0

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                cls_name = names.get(cls_id, "unknown")

                if cls_name not in keep:
                    continue

                xyxy = box.xyxy[0].tolist()

                # --- Forensic Guardrail: Filter Out-of-Scale Detections ---
                # Filter out boxes that are too small to be real (e.g., < 4x4 pixels)
                # or have extreme aspect ratios (e.g., a 1px wide line)
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                if bw < 4 or bh < 4 or (bw / max(1, bh) > 20) or (bh / max(1, bw) > 20):
                    continue
                # -----------------------------------------------------------

                conf_value = float(box.conf.item())

                detections.append(
                    Detection(
                        class_name=cls_name,
                        label=YOLO_CLASS_MAPPING.get(cls_name, cls_name),
                        confidence=conf_value,
                        bbox=BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                        source=source_tag,
                    )
                )

        annotated = draw_detections(image, detections)
        return self._build_result(image, source_name, detections, raw_count, annotated, source_tag)

    def detect_image_raw(
        self,
        image: Image.Image,
        source_name: str = "image",
        confidence: float | None = None,
        iou: float | None = None,
        source_tag: str = "general",
        imgsz: int | None = None,
    ) -> list[Detection]:
        """
        Run inference and return raw `Detection` objects (no drawing).
        Used by `MultiSourceDetector` to merge results across models.
        """
        conf = self.confidence if confidence is None else confidence
        iou_thr = self.iou if iou is None else iou

        def _predict() -> object:
            predict_kwargs: dict[str, Any] = {
                "source": image,
                "conf": conf,
                "iou": iou_thr,
                "verbose": False,
            }
            if imgsz is not None:
                predict_kwargs["imgsz"] = imgsz
            return self.model.predict(**predict_kwargs)

        try:
            results = retry(
                _predict,
                attempts=self.retry_attempts,
                initial_delay=self.retry_initial_delay,
                exceptions=(RuntimeError, OSError),
            )
        except (RuntimeError, OSError):
            raise

        out: list[Detection] = []
        if not results:
            return out
        result = results[0]
        names = result.names
        if result.boxes is None:
            return out
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id, "unknown")

            # Filter out irrelevant COCO classes (e.g. skateboard, donut, bench) for general model passes
            if source_tag in {"general", "weapon-scan"} and cls_name not in RELEVANT_CLASSES:
                continue

            xyxy = box.xyxy[0].tolist()
            lbl = YOLO_CLASS_MAPPING.get(cls_name, YOLO_CLASS_MAPPING.get(cls_name.lower(), cls_name.lower()))
            out.append(
                Detection(
                    class_name=cls_name,
                    label=lbl,
                    confidence=float(box.conf.item()),
                    bbox=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                    source=source_tag,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_result(
        self,
        image: Image.Image,
        source_name: str,
        detections: list[Detection],
        raw_count: int,
        annotated: Image.Image | None = None,
        source_tag: str = "general",
    ) -> DetectionResult:
        if annotated is None:
            annotated = draw_detections(image, detections)
        return DetectionResult(
            source_name=source_name,
            timestamp=datetime.now(),
            detections=detections,
            raw_count=raw_count,
            annotated_image=annotated,
            model_name=self.model_name,
            source_tag=source_tag,
        )


# ----------------------------------------------------------------------
# Multi-source detector (general + optional weapon model)
# ----------------------------------------------------------------------
class MultiSourceDetector:
    """
    Combines a general-purpose YOLODetector with an optional weapon
    detector and merges their detections.

    The general detector runs twice:
    1. with the default confidence and `RELEVANT_CLASSES` (persons,
       vehicles, bags, plus the COCO weapon classes).
    2. with a lower confidence and only the COCO weapon classes
       (knife, baseball bat, scissors, bottle) — the "weapon scan".

    If `WEAPON_MODEL_NAME` (default: `models/weapon.pt`) exists, a
    second YOLODetector is loaded with `WEAPON_CONF_THRESHOLD` and
    its detections are merged in (this is the only way to add gun
    detection — COCO has no gun class).

    The merger is a per-class NMS so duplicates from the two COCO
    runs of the same model are dropped.
    """

    def __init__(
        self,
        general: YOLODetector | None = None,
        weapon_model_path: str | Path | None = None,
        iou_threshold: float = CROSS_MODEL_IOU_THRESHOLD,
    ) -> None:
        self.general = general or YOLODetector()
        self.iou_threshold = float(iou_threshold)
        self._weapon: YOLODetector | None = None
        self._weapon_attempted_path: str | None = None
        self._threat_weapon: YOLODetector | None = None
        self._threat_weapon_attempted_path: str | None = None
        if weapon_model_path is not None:
            self._maybe_init_weapon(Path(weapon_model_path))
        else:
            w_path = self._weapon_model_path()
            if w_path is not None:
                self._maybe_init_weapon(w_path)
            t_path = self._threat_weapon_model_path()
            if t_path is not None:
                self._maybe_init_threat_weapon(t_path)

    @property
    def weapon_model_loaded(self) -> bool:
        return self._weapon is not None

    @property
    def threat_weapon_model_loaded(self) -> bool:
        return self._threat_weapon is not None

    def _weapon_model_path(self) -> Path | None:
        # Locate the optional weapon model. Returns None when it's
        # not present so the caller can render the limit honestly.
        candidates = [Path(WEAPON_MODEL_NAME)]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _threat_weapon_model_path(self) -> Path | None:
        """Locate the optional second weapon model (`models/threat_weapon.pt`).

        This model adds `knife` + `explosion` + `grenade` on top of the
        classes `weapon.pt` covers (`Gun`). Loaded only when the file
        is actually present — never fabricated.
        """
        return self._resolve_existing("models/threat_weapon.pt")

    @staticmethod
    def _resolve_existing(path_like: str | Path) -> Path | None:
        candidate = Path(path_like)
        return candidate if candidate.exists() else None

    def _maybe_init_weapon(self, path: Path) -> None:
        self._weapon_attempted_path = str(path)
        if not path.exists():
            self._weapon = None
            return
        try:
            self._weapon = YOLODetector(
                model_name=str(path),
                confidence=WEAPON_CONF_THRESHOLD,
                iou=self.general.iou,
            )
        except (RuntimeError, OSError, ValueError):
            self._weapon = None

    def _maybe_init_threat_weapon(self, path: Path) -> None:
        """Initialize the broader threat weapon model (knife + explosion + grenade
        + Gun). Same defensive behaviour as `_maybe_init_weapon`."""
        self._threat_weapon_attempted_path = str(path)
        if not path.exists():
            self._threat_weapon = None
            return
        try:
            self._threat_weapon = YOLODetector(
                model_name=str(path),
                confidence=WEAPON_CONF_THRESHOLD,
                iou=self.general.iou,
            )
        except (RuntimeError, OSError, ValueError):
            self._threat_weapon = None

    def detect_image(
        self,
        image: Image.Image,
        source_name: str = "image",
        confidence: float | None = None,
        iou: float | None = None,
    ) -> DetectionResult:
        """
        Run the full multi-source pipeline on a single image.

        Returns a `DetectionResult` whose `detections` is the merged
        list and whose `model_name` lists every model that fired.
        """
        conf = self.general.confidence if confidence is None else confidence
        iou_thr = self.general.iou if iou is None else iou

        # Run 1: general model with full RELEVANT_CLASSES.
        general_raw = self.general.detect_image_raw(
            image, source_name=source_name,
            confidence=conf, iou=iou_thr, source_tag="general",
        )

        # Run 2: general model with a lower conf, COCO weapon classes only.
        weapon_scan_raw = self.general.detect_image_raw(
            image, source_name=source_name,
            confidence=WEAPON_SCAN_CONF_THRESHOLD,
            iou=iou_thr,
            source_tag="weapon-scan",
        )
        # Keep only the COCO weapon classes from the second pass.
        weapon_scan_raw = [
            d for d in weapon_scan_raw
            if d.class_name in COCO_WEAPON_CLASSES
        ]

        # Dedicated weapon model: Pass 1 (high-res RGB) + Pass 2 (conditional CLAHE contrast-enhanced)
        weapon_extra: list[Detection] = []
        if self._weapon is None:
            path = self._weapon_model_path()
            if path is not None and self._weapon_attempted_path != str(path):
                self._maybe_init_weapon(path)
        if self._weapon is not None:
            w_conf = WEAPON_CONF_THRESHOLD if confidence is None else min(confidence, WEAPON_CONF_THRESHOLD)
            pass1: list[Detection] = []
            pass2: list[Detection] = []
            try:
                pass1 = self._weapon.detect_image_raw(
                    image, source_name=source_name,
                    confidence=w_conf,
                    iou=iou_thr,
                    source_tag="weapon",
                    imgsz=1280,
                )
            except (RuntimeError, OSError):
                pass1 = []

            # Only run CLAHE pass if Pass 1 didn't yield a high-confidence weapon detection
            if not any(d.confidence >= 0.70 for d in pass1):
                try:
                    enhanced_img = _enhance_contrast(image)
                    pass2 = self._weapon.detect_image_raw(
                        enhanced_img, source_name=source_name,
                        confidence=w_conf,
                        iou=iou_thr,
                        source_tag="weapon",
                        imgsz=1280,
                    )
                except (RuntimeError, OSError):
                    pass2 = []

            W, H = image.size
            for d in pass1 + pass2:
                bw = d.bbox.x2 - d.bbox.x1
                bh = d.bbox.y2 - d.bbox.y1
                rel_area = (bw * bh) / max(1.0, float(W * H))
                aspect = bw / max(1.0, bh)
                if bw >= 8 and bh >= 8 and (rel_area >= 0.0003 or d.confidence >= 0.80) and (0.12 < aspect < 8.0 or d.confidence >= 0.80):
                    if bw * bh < 0.50 * W * H:
                        weapon_extra.append(d)

        # Broader threat weapon model (knife + explosion + grenade + Gun).
        threat_extra: list[Detection] = []
        if self._threat_weapon is None:
            t_path = self._threat_weapon_model_path()
            if t_path is not None and self._threat_weapon_attempted_path != str(t_path):
                self._maybe_init_threat_weapon(t_path)
        if self._threat_weapon is not None:
            t_conf = WEAPON_CONF_THRESHOLD if confidence is None else min(confidence, WEAPON_CONF_THRESHOLD)
            t_pass1: list[Detection] = []
            t_pass2: list[Detection] = []
            try:
                t_pass1 = self._threat_weapon.detect_image_raw(
                    image, source_name=source_name,
                    confidence=t_conf, iou=iou_thr,
                    source_tag="threat-weapon", imgsz=1280,
                )
            except (RuntimeError, OSError):
                t_pass1 = []

            if not any(d.confidence >= 0.70 for d in t_pass1):
                try:
                    t_enhanced = _enhance_contrast(image)
                    t_pass2 = self._threat_weapon.detect_image_raw(
                        t_enhanced, source_name=source_name,
                        confidence=t_conf, iou=iou_thr,
                        source_tag="threat-weapon", imgsz=1280,
                    )
                except (RuntimeError, OSError):
                    t_pass2 = []

            W, H = image.size
            for d in t_pass1 + t_pass2:
                bw = d.bbox.x2 - d.bbox.x1
                bh = d.bbox.y2 - d.bbox.y1
                rel_area = (bw * bh) / max(1.0, float(W * H))
                aspect = bw / max(1.0, bh)
                if bw >= 8 and bh >= 8 and (rel_area >= 0.0003 or d.confidence >= 0.80) and (0.12 < aspect < 8.0 or d.confidence >= 0.80):
                    if bw * bh < 0.50 * W * H:
                        threat_extra.append(d)

        merged = _cross_model_nms(
            general_raw + weapon_scan_raw + weapon_extra + threat_extra,
            iou_threshold=self.iou_threshold,
        )

        # Identify which models contributed for the UI layer.
        models_used = sorted({d.source for d in merged if d.source})
        model_name = "general"
        if (
            (self._weapon is not None and any(d.source == "weapon" for d in merged))
            or (self._threat_weapon is not None and any(d.source == "threat-weapon" for d in merged))
        ):
            model_name = "general+weapon"
        elif any(d.source == "weapon-scan" for d in merged):
            model_name = "general (weapon-scan)"

        annotated = draw_detections(image, merged)
        # Phase 46 — apply the two-stage weapon verification so each
        # detection knows whether it is verified, candidate, or
        # non-weapon.
        verified = _verify_detections(merged)
        return DetectionResult(
            source_name=source_name,
            timestamp=datetime.now(),
            detections=verified,
            raw_count=len(general_raw) + len(weapon_scan_raw) + len(weapon_extra) + len(threat_extra),
            annotated_image=draw_detections(image, verified),
            model_name=model_name,
            source_tag="multi",
            models_used=models_used,
        )


# ----------------------------------------------------------------------
# Streamlit-friendly singleton loader
# ----------------------------------------------------------------------
_MULTI_SOURCE_SINGLETON: MultiSourceDetector | None = None


def get_detector(
    model_name: str = YOLO_MODEL_NAME,
    confidence: float = YOLO_CONFIDENCE_THRESHOLD,
    iou: float = YOLO_IOU_THRESHOLD,
) -> YOLODetector:
    """Return a YOLODetector instance."""
    return YOLODetector(model_name=model_name, confidence=confidence, iou=iou)


def get_multi_source_detector() -> MultiSourceDetector:
    """Return a cached singleton MultiSourceDetector."""
    global _MULTI_SOURCE_SINGLETON
    if _MULTI_SOURCE_SINGLETON is None:
        _MULTI_SOURCE_SINGLETON = MultiSourceDetector()
    return _MULTI_SOURCE_SINGLETON


def save_annotated_image(
    result: DetectionResult,
    output_dir: Path,
    file_stem: str,
) -> Path:
    """Save the annotated image to disk and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{file_stem}_annotated.jpg"
    if result.annotated_image is not None:
        result.annotated_image.save(out_path, format="JPEG", quality=90)
    return out_path


def result_to_bytes(result: DetectionResult) -> bytes:
    """Encode the annotated image to JPEG bytes for Streamlit download."""
    if result.annotated_image is None:
        return b""
    buf = BytesIO()
    result.annotated_image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
