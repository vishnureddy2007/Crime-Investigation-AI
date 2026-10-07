"""Tests for ``utils.i18n``."""

from __future__ import annotations

from utils.i18n import (
    DEFAULT_LANGUAGE,
    available_languages,
    is_supported,
    set_language,
    tr,
)


def test_default_language_is_english() -> None:
    assert DEFAULT_LANGUAGE == "en"


def test_available_languages_returns_tuple() -> None:
    langs = available_languages()
    assert isinstance(langs, tuple)
    assert "en" in langs
    assert "es" in langs
    assert "hi" in langs


def test_tr_returns_english_by_default() -> None:
    assert tr("nav.home") == "Home"
    assert tr("common.refresh") == "Refresh"


def test_tr_returns_spanish_when_lang_es() -> None:
    assert tr("nav.home", "es") == "Inicio"


def test_tr_returns_hindi_when_lang_hi() -> None:
    assert tr("nav.home", "hi") == "मुख्य पृष्ठ"


def test_tr_falls_back_to_english_on_missing() -> None:
    # Use a key that exists in English but not (intentionally) in another
    # language by patching TRANSLATIONS via monkeypatch.
    from utils import i18n

    original = i18n.TRANSLATIONS["es"]
    i18n.TRANSLATIONS["es"] = {k: v for k, v in original.items() if k != "common.refresh"}
    try:
        assert tr("common.refresh", "es") == "Refresh"
    finally:
        i18n.TRANSLATIONS["es"] = original


def test_tr_returns_key_on_total_miss() -> None:
    assert tr("does.not.exist") == "does.not.exist"


def test_set_language_rejects_unknown() -> None:
    with __import__("pytest").raises(ValueError):
        set_language("fr")


def test_is_supported() -> None:
    assert is_supported("en")
    assert is_supported("hi")
    assert not is_supported("fr")