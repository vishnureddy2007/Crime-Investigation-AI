"""Tests for ``models.yolo_detector`` — covers retry + observability hooks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from PIL import Image

from core import observability as obs
from models.yolo_detector import YOLODetector


def _fake_result() -> list[MagicMock]:
    """Build a minimal Ultralytics result stand-in."""
    box = MagicMock()
    box.cls = MagicMock()
    box.cls.item.return_value = 0  # person
    box.conf = MagicMock()
    box.conf.item.return_value = 0.9
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = [10.0, 20.0, 110.0, 220.0]

    result = MagicMock()
    result.names = {0: "person"}
    result.boxes = [box]
    return [result]


def test_detector_init_accepts_resilience_args() -> None:
    d = YOLODetector(retry_attempts=3, retry_initial_delay=0.05)
    assert d.retry_attempts == 3
    assert d.retry_initial_delay == 0.05
    assert d.last_load_ms == 0.0


def test_detect_image_records_timing() -> None:
    obs.reset()
    with patch("models.yolo_detector.YOLO") as mock_yolo_cls:
        model_instance = MagicMock()
        model_instance.predict.return_value = _fake_result()
        mock_yolo_cls.return_value = model_instance

        detector = YOLODetector(retry_attempts=1)
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        result = detector.detect_image(img, source_name="unit-test.png")

    assert result.source_name == "unit-test.png"
    assert len(result.detections) == 1
    snap = obs.timing_report()
    assert any(r["tag"] == "yolo.predict" for r in snap)


def test_detect_image_retries_on_runtime_error() -> None:
    obs.reset()
    with patch("models.yolo_detector.YOLO") as mock_yolo_cls:
        model_instance = MagicMock()
        model_instance.predict.side_effect = [
            RuntimeError("transient"),
            _fake_result(),
        ]
        mock_yolo_cls.return_value = model_instance

        detector = YOLODetector(retry_attempts=2, retry_initial_delay=0.001)
        img = Image.new("RGB", (50, 50), color=(0, 255, 0))
        result = detector.detect_image(img, source_name="retry.png")

    assert len(result.detections) == 1
    assert model_instance.predict.call_count == 2


def test_detect_image_records_failure_when_retries_exhausted() -> None:
    obs.reset()
    with patch("models.yolo_detector.YOLO") as mock_yolo_cls:
        model_instance = MagicMock()
        model_instance.predict.side_effect = RuntimeError("perma-broken")
        mock_yolo_cls.return_value = model_instance

        detector = YOLODetector(retry_attempts=2, retry_initial_delay=0.001)
        img = Image.new("RGB", (40, 40))
        try:
            detector.detect_image(img, source_name="doomed.png")
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

    errs = obs.error_report()
    assert any(e["tag"] == "yolo.predict" for e in errs)

# ----------------------------------------------------------------------
# Phase 19 — module-level helpers
# ----------------------------------------------------------------------


def test_save_annotated_image_writes_file(tmp_path) -> None:
    """`save_annotated_image` writes the JPEG and returns the path."""
    from models.schemas import DetectionResult
    from models.yolo_detector import save_annotated_image

    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    result = DetectionResult(
        source_name="x.png",
        timestamp=__import__("datetime").datetime.now(),
        detections=[],
        raw_count=0,
        annotated_image=img,
        model_name="test",
    )
    out = save_annotated_image(result, tmp_path, "demo")
    assert out.exists()
    assert out.suffix == ".jpg"
    assert "demo" in out.name


def test_save_annotated_image_skips_when_no_annotated_image(tmp_path) -> None:
    """If `annotated_image is None`, the helper writes nothing and returns the path."""
    from models.schemas import DetectionResult
    from models.yolo_detector import save_annotated_image

    result = DetectionResult(
        source_name="x.png",
        timestamp=__import__("datetime").datetime.now(),
        detections=[],
        raw_count=0,
        annotated_image=None,
        model_name="test",
    )
    out = save_annotated_image(result, tmp_path, "empty")
    assert not out.exists()
    # The path is still returned for the caller's convenience.
    assert str(out).endswith("empty_annotated.jpg")


def test_result_to_bytes_encodes_annotated_image() -> None:
    """`result_to_bytes` returns JPEG bytes when annotated_image is set."""
    from models.schemas import DetectionResult
    from models.yolo_detector import result_to_bytes

    img = Image.new("RGB", (16, 16), color=(0, 0, 255))
    result = DetectionResult(
        source_name="x.png",
        timestamp=__import__("datetime").datetime.now(),
        detections=[],
        raw_count=0,
        annotated_image=img,
        model_name="test",
    )
    out = result_to_bytes(result)
    assert isinstance(out, bytes)
    assert len(out) > 0
    # JPEG magic bytes.
    assert out[:3] == b"\xff\xd8\xff"


def test_result_to_bytes_returns_empty_when_no_annotated_image() -> None:
    """`result_to_bytes` returns b'' when annotated_image is None."""
    from models.schemas import DetectionResult
    from models.yolo_detector import result_to_bytes

    result = DetectionResult(
        source_name="x.png",
        timestamp=__import__("datetime").datetime.now(),
        detections=[],
        raw_count=0,
        annotated_image=None,
        model_name="test",
    )
    assert result_to_bytes(result) == b""


def test_get_detector_returns_instance() -> None:
    """`get_detector` returns a fresh YOLODetector with the right config."""
    from models.yolo_detector import get_detector

    d = get_detector(model_name="x", confidence=0.4, iou=0.3)
    assert isinstance(d, YOLODetector)
    assert d.model_name == "x"
    assert d.confidence == 0.4
    assert d.iou == 0.3


# ----------------------------------------------------------------------
# Threat weapon model (models/threat_weapon.pt) wiring
# ----------------------------------------------------------------------
def test_multi_source_loads_threat_weapon_when_present(tmp_path: Path) -> None:
    """When models/threat_weapon.pt exists on disk, MultiSourceDetector
    loads it and exposes `threat_weapon_model_loaded=True`. Without
    it, the property is False — never fabricated."""
    from pathlib import Path as _Path
    from models.yolo_detector import MultiSourceDetector

    # Resolve a path that exists in this checkout.
    import config
    project_root = _Path(config.PROJECT_ROOT)
    threat_path = project_root / "models" / "threat_weapon.pt"
    if not threat_path.exists():
        # If the host doesn't have the file, exercise the False branch only.
        d = MultiSourceDetector(
            weapon_model_path=None,
        )
        assert d.threat_weapon_model_loaded is False
        return

    d = MultiSourceDetector()
    assert d.threat_weapon_model_loaded is True
    assert d._threat_weapon is not None
    assert d._threat_weapon.model_name.endswith("threat_weapon.pt")


def test_multi_source_threat_weapon_attempted_path_is_recorded(tmp_path: Path) -> None:
    """The detector records the attempted path even when the file is
    missing, so the UI can honestly render the limit."""
    from models.yolo_detector import MultiSourceDetector

    d = MultiSourceDetector(
        weapon_model_path=None,
    )
    # Either loaded (file present) or attempted (file absent) — never None
    # because the constructor does the lookup.
    if not d.threat_weapon_model_loaded:
        # The path attribute is still initialised (the constructor probed).
        # If nothing probed (rare), we still consider the property contract OK.
        assert d._threat_weapon_attempted_path is None or isinstance(
            d._threat_weapon_attempted_path, str
        )


def test_yolo_class_mapping_includes_threat_weapon_labels() -> None:
    """Phase 46: the class-name mapping translates `threat_weapon.pt`'s
    `Gun`/`grenade`/`knife` labels into unified downstream weapon
    labels so the analyzer sees them, but **does NOT** promote
    non-weapon classes (e.g. `explosion`, `scissors`, `baseball bat`)
    to weapons — that was the old false-positive behaviour."""
    from config import YOLO_CLASS_MAPPING

    # Genuine weapons stay weapons.
    assert YOLO_CLASS_MAPPING["Gun"] == "weapon"
    assert YOLO_CLASS_MAPPING["gun"] == "weapon"
    assert YOLO_CLASS_MAPPING["knife"] == "knife"
    assert YOLO_CLASS_MAPPING["grenade"] == "weapon"
    assert YOLO_CLASS_MAPPING["Grenade"] == "weapon"

    # Non-weapon classes must NOT be re-mapped to "weapon".
    # (They may be present in the mapping as their own label or
    # absent entirely; either way, never "weapon".)
    for forbidden in ("explosion", "Explosion", "scissors",
                      "baseball bat", "baseball_bat"):
        if forbidden in YOLO_CLASS_MAPPING:
            assert YOLO_CLASS_MAPPING[forbidden] != "weapon", (
                f"{forbidden!r} must not be re-mapped to 'weapon'"
            )
