"""Unit tests for `utils/face_detection.py`."""
from __future__ import annotations

from PIL import Image

from utils.face_detection import FaceBox, detect_faces, is_face_detection_available


def _image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(100, 100, 100))


def test_is_face_detection_available_returns_bool() -> None:
    assert isinstance(is_face_detection_available(), bool)


def test_detect_faces_returns_list_of_boxes() -> None:
    boxes = detect_faces(_image())
    assert isinstance(boxes, list)
    for b in boxes:
        assert isinstance(b, FaceBox)
        assert b.x2 >= b.x1
        assert b.y2 >= b.y1
        assert 0.0 <= b.confidence <= 1.0


def test_detect_faces_never_raises() -> None:
    boxes = detect_faces(None)  # type: ignore[arg-type]
    assert isinstance(boxes, list)


def test_as_dict_shape() -> None:
    b = FaceBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9)
    d = b.as_dict()
    assert d["x1"] == 0
    assert d["confidence"] == 0.9

# ----------------------------------------------------------------------
# Phase 17 — coverage with mocked cv2
# ----------------------------------------------------------------------


def test_detect_faces_uses_cascade_when_available(monkeypatch, tmp_path) -> None:
    """If cv2 + cascade are available, the detector returns FaceBoxes."""
    import sys
    import types

    fake_cv2 = types.ModuleType("cv2")

    class _Classifier:
        def detectMultiScale(self, _gray, scaleFactor, minNeighbors, minSize):
            # Return two boxes.
            return [(0, 0, 30, 30), (50, 50, 20, 20)]

    fake_cv2.data = types.SimpleNamespace(haarcascades=str(tmp_path))
    fake_cv2.CascadeClassifier = lambda _path: _Classifier()
    fake_cv2.cvtColor = lambda arr, _code: arr  # pass-through
    fake_cv2.COLOR_RGB2GRAY = 0
    (tmp_path / "haarcascade_frontalface_default.xml").write_text("<fake/>")
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    from utils.face_detection import detect_faces

    boxes = detect_faces(_image())
    assert len(boxes) == 2
    assert boxes[0].x1 == 0
    assert boxes[0].x2 == 30


def test_detect_faces_returns_empty_when_cascade_missing(monkeypatch, tmp_path) -> None:
    """If the cascade XML is absent, detect_faces returns [] gracefully."""
    import sys
    import types

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.data = types.SimpleNamespace(haarcascades=str(tmp_path))
    # No XML file written.
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    from utils.face_detection import detect_faces, is_face_detection_available

    assert is_face_detection_available() is False
    assert detect_faces(_image()) == []


def test_detect_faces_swallows_cv2_runtime_error(monkeypatch, tmp_path) -> None:
    """A cv2 exception must be swallowed, not raised."""
    import sys
    import types

    fake_cv2 = types.ModuleType("cv2")

    def _boom(*_a, **_kw):
        raise RuntimeError("cv2 broken")

    fake_cv2.data = types.SimpleNamespace(haarcascades=str(tmp_path))
    fake_cv2.cvtColor = _boom
    fake_cv2.COLOR_RGB2GRAY = 0
    fake_cv2.CascadeClassifier = _boom
    (tmp_path / "haarcascade_frontalface_default.xml").write_text("<fake/>")
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    from utils.face_detection import detect_faces

    assert detect_faces(_image()) == []


# ----------------------------------------------------------------------
# Phase 37 — coverage push to 100%
# ----------------------------------------------------------------------


def test_detect_faces_accepts_numpy_array_input(monkeypatch, tmp_path) -> None:
    """detect_faces() must accept a numpy array (no .convert method) and
    route it through the array branch (utils/face_detection.py:73)."""
    import sys
    import types
    import numpy as np

    fake_cv2 = types.ModuleType("cv2")

    class _Classifier:
        def detectMultiScale(self, _gray, scaleFactor, minNeighbors, minSize):
            return [(5, 5, 20, 20)]

    fake_cv2.data = types.SimpleNamespace(haarcascades=str(tmp_path))
    fake_cv2.CascadeClassifier = lambda _p: _Classifier()
    fake_cv2.cvtColor = lambda arr, _code: arr
    fake_cv2.COLOR_RGB2GRAY = 0
    (tmp_path / "haarcascade_frontalface_default.xml").write_text("<fake/>")
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    from utils.face_detection import detect_faces

    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = detect_faces(arr)
    assert isinstance(boxes, list)
    assert len(boxes) == 1
    assert boxes[0].x1 == 5
    assert boxes[0].y2 == 25  # y + h


def test_cascade_path_returns_none_when_cv2_raises(monkeypatch) -> None:
    """If cv2.data.haarcascades access raises OSError (utils/face_detection.py:50-51),
    _cascade_path() returns None and is_face_detection_available() reports False."""
    import sys
    import types

    fake_cv2 = types.ModuleType("cv2")

    class _BrokenAttr:
        def __getattr__(self, _name):
            raise OSError("cv2.data unavailable (test stub)")

    fake_cv2.data = _BrokenAttr()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    from utils.face_detection import (
        detect_faces,
        is_face_detection_available,
    )

    assert is_face_detection_available() is False
    assert detect_faces(_image()) == []
