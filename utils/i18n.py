"""
Lightweight in-app translation helper.

Pure-Python, no external deps. Used by the **Settings** page to swap the
UI between English and a couple of bundled locales (Spanish, Hindi).

Public surface
--------------
- ``DEFAULT_LANGUAGE`` — fall-back when the session key isn't set.
- ``available_languages()`` — list of bundled locale codes.
- ``tr(key, lang=DEFAULT_LANGUAGE)`` — return the localised string, or
  ``key`` itself when the lookup misses.
- ``set_language(lang)`` — convenience used by tests.

This is intentionally minimal — full gettext isn't needed for two
strings per page. If a third locale lands, swap to ``gettext``.
"""

from __future__ import annotations

from typing import Mapping


DEFAULT_LANGUAGE: str = "en"

# Lookup table — keep entries short. ``en`` keys are the canonical IDs.
TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "app.title": "AI Crime Investigation Assistant",
        "app.tagline": "Detect. Analyze. Reconstruct. Report.",
        "nav.home": "Home",
        "nav.dashboard": "Dashboard",
        "nav.detect": "Image Detection",
        "nav.video": "Video Processing",
        "nav.evidence": "Evidence Analysis",
        "nav.summary": "AI Summary",
        "nav.chat": "Chat Assistant",
        "nav.timeline": "Crime Timeline",
        "nav.predict": "Crime Prediction",
        "nav.report": "Report",
        "nav.reconstruct": "Reconstruction",
        "nav.cases": "Case History",
        "nav.analytics": "Analytics",
        "nav.settings": "Settings",
        "nav.help": "Help",
        "nav.contact": "Contact",
        "nav.about": "About",
        "nav.status": "System Status",
        "common.export": "Export",
        "common.refresh": "Refresh",
        "common.cancel": "Cancel",
        "common.confirm": "Confirm",
        "common.loading": "Loading…",
    },
    "es": {
        "app.title": "Asistente de Investigación Criminal con IA",
        "app.tagline": "Detectar. Analizar. Reconstruir. Informar.",
        "nav.home": "Inicio",
        "nav.dashboard": "Panel",
        "nav.detect": "Detección de Imágenes",
        "nav.video": "Procesamiento de Vídeo",
        "nav.evidence": "Análisis de Evidencias",
        "nav.summary": "Resumen de IA",
        "nav.chat": "Asistente de Chat",
        "nav.timeline": "Línea de Tiempo",
        "nav.predict": "Predicción",
        "nav.report": "Informe",
        "nav.reconstruct": "Reconstrucción",
        "nav.cases": "Historial",
        "nav.analytics": "Analítica",
        "nav.settings": "Ajustes",
        "nav.help": "Ayuda",
        "nav.contact": "Contacto",
        "nav.about": "Acerca de",
        "nav.status": "Estado del Sistema",
        "common.export": "Exportar",
        "common.refresh": "Actualizar",
        "common.cancel": "Cancelar",
        "common.confirm": "Confirmar",
        "common.loading": "Cargando…",
    },
    "hi": {
        "app.title": "एआई अपराध जाँच सहायक",
        "app.tagline": "पहचानें। विश्लेषण करें। पुनर्निर्माण करें। रिपोर्ट करें।",
        "nav.home": "मुख्य पृष्ठ",
        "nav.dashboard": "डैशबोर्ड",
        "nav.detect": "छवि पहचान",
        "nav.video": "वीडियो प्रसंस्करण",
        "nav.evidence": "साक्ष्य विश्लेषण",
        "nav.summary": "एआई सारांश",
        "nav.chat": "चैट सहायक",
        "nav.timeline": "घटना क्रम",
        "nav.predict": "भविष्यवाणी",
        "nav.report": "रिपोर्ट",
        "nav.reconstruct": "पुनर्निर्माण",
        "nav.cases": "केस इतिहास",
        "nav.analytics": "विश्लेषण",
        "nav.settings": "सेटिंग्स",
        "nav.help": "सहायता",
        "nav.contact": "संपर्क",
        "nav.about": "के बारे में",
        "nav.status": "सिस्टम स्थिति",
        "common.export": "निर्यात",
        "common.refresh": "ताज़ा करें",
        "common.cancel": "रद्द करें",
        "common.confirm": "पुष्टि करें",
        "common.loading": "लोड हो रहा है…",
    },
}


def available_languages() -> tuple[str, ...]:
    """Return the tuple of bundled language codes."""
    return tuple(TRANSLATIONS.keys())


def _lookup(key: str, lang: str) -> str | None:
    table: Mapping[str, str] = TRANSLATIONS.get(lang, {})
    return table.get(key)


def tr(key: str, lang: str = DEFAULT_LANGUAGE) -> str:
    """Return the localised string for ``key``.

    Falls back to English, then to the key itself when the lookup
    misses in both.
    """
    value = _lookup(key, lang)
    if value is not None:
        return value
    if lang != DEFAULT_LANGUAGE:
        value = _lookup(key, DEFAULT_LANGUAGE)
        if value is not None:
            return value
    return key


def set_language(lang: str) -> None:
    """Validate a language code; raises :class:`ValueError` if unsupported."""
    if lang not in TRANSLATIONS:
        raise ValueError(f"unsupported language: {lang}")


def is_supported(lang: str) -> bool:
    """Return True if ``lang`` is one of the bundled locales."""
    return lang in TRANSLATIONS