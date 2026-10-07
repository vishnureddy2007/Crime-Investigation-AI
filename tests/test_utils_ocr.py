"""Unit tests for `utils/ocr.py`."""
from __future__ import annotations

from PIL import Image

from utils.ocr import extract_text, is_ocr_available, OcrResult


def _image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(255, 255, 255))


def test_is_ocr_available_returns_bool() -> None:
    assert isinstance(is_ocr_available(), bool)


def test_extract_text_returns_ocr_result() -> None:
    r = extract_text(_image())
    assert isinstance(r, OcrResult)
    assert r.engine in ("tesseract", "stub")
    assert 0.0 <= r.confidence <= 1.0


def test_extract_text_never_raises() -> None:
    # Pass junk; the stub path must catch everything.
    r = extract_text(None)  # type: ignore[arg-type]
    assert isinstance(r, OcrResult)


def test_as_dict_shape() -> None:
    r = extract_text(_image())
    d = r.as_dict()
    assert {"text", "confidence", "engine"} <= d.keys()


def test_stub_is_deterministic() -> None:
    a = extract_text(_image())
    b = extract_text(_image())
    # Stub path uses sha1 of the byte buffer — same image → same conf.
    if a.engine == "stub" and b.engine == "stub":
        assert a.confidence == b.confidence

# ----------------------------------------------------------------------
# Phase 17 — coverage with mocked pytesseract
# ----------------------------------------------------------------------


def test_extract_text_uses_tesseract_when_available(monkeypatch) -> None:
    """If pytesseract is importable, extract_text uses it."""
    import sys
    import types

    fake = types.ModuleType("pytesseract")

    class _Output:
        DICT = "dict"

    fake.image_to_string = lambda _img: "Hello World"
    fake.image_to_data = lambda _img, output_type=None: {"conf": [80, 90]}
    fake.Output = _Output
    monkeypatch.setitem(sys.modules, "pytesseract", fake)

    from utils.ocr import extract_text

    r = extract_text(_image())
    assert r.engine == "tesseract"
    assert r.text == "Hello World"
    assert r.confidence > 0


def test_extract_text_falls_back_on_tesseract_error(monkeypatch) -> None:
    """A runtime error from pytesseract falls back to the stub."""
    import sys
    import types

    fake = types.ModuleType("pytesseract")

    class _Output:
        DICT = "dict"

    def _boom(*_a, **_kw):
        raise RuntimeError("tesseract crashed")

    fake.image_to_string = _boom
    fake.image_to_data = _boom
    fake.Output = _Output
    monkeypatch.setitem(sys.modules, "pytesseract", fake)

    from utils.ocr import extract_text

    r = extract_text(_image())
    assert r.engine == "stub"
    assert isinstance(r.text, str)


def test_extract_text_handles_garbage_conf_data(monkeypatch) -> None:
    """If `conf` is junk, the average stays in [0, 1]."""
    import sys
    import types

    fake = types.ModuleType("pytesseract")

    class _Output:
        DICT = "dict"

    fake.image_to_string = lambda _img: "x"
    fake.image_to_data = lambda _img, output_type=None: {"conf": ["abc", -1, 50, 150]}
    fake.Output = _Output
    monkeypatch.setitem(sys.modules, "pytesseract", fake)

    from utils.ocr import extract_text

    r = extract_text(_image())
    assert r.engine == "tesseract"
    # Only 50 was valid; avg = 50 / 100.
    assert 0.0 <= r.confidence <= 1.0


def test_stub_extract_falls_back_to_zero_conf_on_tobytes_error(monkeypatch) -> None:
    """If the input image is so broken that tobytes() raises, _stub_extract
    catches the exception and returns confidence=0.0 with engine='stub'.
    Line 70-71 of utils/ocr.py."""
    from utils.ocr import _stub_extract

    class _BrokenImage:
        # Enter the PIL branch: has size and convert, but tobytes raises.
        size = (10, 10)

        def convert(self, _mode):
            return self

        def thumbnail(self, _size):
            pass

        def tobytes(self):
            raise RuntimeError("tobytes exploded")

    result = _stub_extract(_BrokenImage())
    assert result.confidence == 0.0
    assert result.engine == "stub"


def test_extract_text_data_conf_data_raises(monkeypatch) -> None:
    """When pytesseract is available and image_to_string works, but
    image_to_data raises (e.g. malformed Tesseract output),
    extract_text() returns the text with confidence=0.0.
    Line 96-97 of utils/ocr.py."""
    from utils import ocr as ocr_mod

    class FakePytesseract:
        class Output:
            DICT = "dict"

        @staticmethod
        def image_to_string(image):
            return "hello world"

        @staticmethod
        def image_to_data(image, output_type=None):
            raise RuntimeError("data call failed")

    fake_module = FakePytesseract()
    monkeypatch.setattr(ocr_mod, "pytesseract", None, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "pytesseract", fake_module)
    monkeypatch.setattr(ocr_mod, "is_ocr_available", lambda: True)

    from PIL import Image
    img = Image.new("RGB", (50, 50), (200, 200, 200))
    result = ocr_mod.extract_text(img)
    assert result.text == "hello world"
    assert result.confidence == 0.0
    assert result.engine == "tesseract"


# ----------------------------------------------------------------------
# Phase 38 — coverage push to 100%
# ----------------------------------------------------------------------


def test_is_ocr_available_returns_false_on_attribute_error(monkeypatch) -> None:
    """is_ocr_available()'s AttributeError branch (utils/ocr.py:43-44)
    catches the rare case where importing pytesseract raises
    AttributeError (e.g. on an exotic interpreter build). The helper
    must return False, not crash."""
    import builtins

    # Replace `builtins.__import__` so that `import pytesseract`
    # inside is_ocr_available() raises AttributeError on every call.
    real_import = builtins.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "pytesseract":
            raise AttributeError("pytesseract broken (test stub)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raising_import)

    # Now call the real production function — its `(ImportError,
    # AttributeError)` clause must swallow the AttributeError and
    # return False.
    from utils.ocr import is_ocr_available as _real_is_ocr_available

    assert _real_is_ocr_available() is False


def test_extract_text_short_circuits_to_stub_when_ocr_unavailable(monkeypatch) -> None:
    """extract_text() short-circuits to _stub_extract when
    is_ocr_available() returns False (utils/ocr.py:82). No call to
    pytesseract should be attempted."""
    from utils import ocr as ocr_mod3

    monkeypatch.setattr(ocr_mod3, "is_ocr_available", lambda: False)

    from PIL import Image
    img = Image.new("RGB", (32, 32), (255, 255, 255))
    result = ocr_mod3.extract_text(img)
    assert isinstance(result, OcrResult)
    assert result.engine == "stub"
    assert 0.0 <= result.confidence <= 1.0


def test_extract_text_falls_back_to_stub_on_pytesseract_oserror(monkeypatch) -> None:
    """When pytesseract.image_to_string raises OSError, extract_text()
    falls back to _stub_extract (utils/ocr.py:100)."""
    import sys
    import types

    class _FakePytesseract:
        @staticmethod
        def image_to_string(_image):
            raise OSError("tesseract binary missing (test stub)")

    monkeypatch.setitem(sys.modules, "pytesseract", _FakePytesseract())
    from utils import ocr as ocr_mod4
    monkeypatch.setattr(ocr_mod4, "pytesseract", _FakePytesseract(), raising=False)
    monkeypatch.setattr(ocr_mod4, "is_ocr_available", lambda: True)

    from PIL import Image
    img = Image.new("RGB", (32, 32), (255, 255, 255))
    result = ocr_mod4.extract_text(img)
    assert isinstance(result, OcrResult)
    assert result.engine == "stub"
    assert result.text == ""
    assert 0.0 <= result.confidence <= 1.0


def test_stub_extract_routes_numpy_array_via_tobytes_branch() -> None:
    """_stub_extract() with a numpy array (has .tobytes but no
    .size+.convert) routes through utils/ocr.py:63-64."""
    import numpy as np
    from utils.ocr import _stub_extract

    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    result = _stub_extract(arr)
    assert isinstance(result, OcrResult)
    assert result.engine == "stub"
    # The deterministic hash must be in [0, 1].
    assert 0.0 <= result.confidence <= 1.0


def test_stub_extract_handles_unknown_object_via_str_fallback() -> None:
    """_stub_extract() with an object that has neither .size+.convert
    nor .tobytes routes through the str() fallback (utils/ocr.py:66).
    Examples include a plain Python list."""
    from utils.ocr import _stub_extract

    # A plain list has no .size/.convert and no .tobytes — exactly the
    # fallback case.
    result = _stub_extract([1, 2, 3, 4])
    assert isinstance(result, OcrResult)
    assert result.engine == "stub"
    assert result.text == ""
    assert 0.0 <= result.confidence <= 1.0
