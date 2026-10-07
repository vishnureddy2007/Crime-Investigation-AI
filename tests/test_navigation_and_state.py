"""Tests for navigation registry + centralized session keys."""
from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path


def _load_app_module():
    """Load app.py directly as a module (to avoid namespace collision with app/ directory)."""
    spec = importlib.util.spec_from_file_location("app_entry", Path("app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_route_map() -> dict[str, tuple[str, str]]:
    """Read the route map directly out of app.py source."""
    src = Path("app.py").read_text(encoding="utf-8")
    match = re.search(r"route\s*=\s*\{(.*?)\n\s*\}", src, re.DOTALL)
    assert match, "Could not find route map in app.py"
    block = match.group(1)
    route: dict[str, tuple[str, str]] = {}
    pattern = re.compile(
        r'"([^"]+)"\s*:\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)'
    )
    for m in pattern.finditer(block):
        route[m.group(1)] = (m.group(2), m.group(3))
    return route


def test_app_pages_list_is_complete() -> None:
    """Every page module under pages/ should be reachable via the navigation registry in app.py."""
    pages_dir = Path("pages")
    page_modules = {
        p.stem for p in pages_dir.glob("*.py")
        if p.stem not in {"__init__", "_layout", "_state"}
    }

    app_mod = _load_app_module()
    registered: set[str] = set()
    route = _extract_route_map()
    for module_name, _attr in route.values():
        if module_name.startswith("pages."):
            registered.add(module_name.split(".")[-1])

    missing = page_modules - registered
    assert not missing, f"Unregistered page modules: {missing}"

    for module_name, attr in route.values():
        mod = importlib.import_module(module_name)
        assert hasattr(mod, attr), (
            f"{module_name} missing render attribute '{attr}'"
        )

    assert isinstance(app_mod.PAGES, list) and len(app_mod.PAGES) >= 5


def test_navigation_order_matches_page_list() -> None:
    """The PAGES list in app.py drives the sidebar radio; the route map must contain an entry for every page."""
    app_mod = _load_app_module()
    route = _extract_route_map()
    for page in app_mod.PAGES:
        assert page in route, f"Page '{page}' has no route entry."


def test_app_uses_text_only_page_icon() -> None:
    """The browser tab icon must NOT be an emoji chrome glyph."""
    text = Path("app.py").read_text(encoding="utf-8")
    assert "\\U0001F50D" not in text
    assert "🔍" not in text


def test_session_keys_are_unique() -> None:
    """Every session-state key the app uses must be a single source of truth."""
    from pages._state import SessionKeys

    values = [v for v in vars(SessionKeys).values() if isinstance(v, str)]
    assert len(values) == len(set(values)), (
        "Duplicate session-state keys in SessionKeys: "
        f"{[v for v in values if values.count(v) > 1]}"
    )


def test_session_keys_match_expected_strings() -> None:
    """The SessionKeys values are part of the persistence contract."""
    from pages._state import SessionKeys

    assert SessionKeys.LAST_ANALYSIS == "last_analysis"
    assert SessionKeys.LAST_BATCH_RESULT == "last_batch_result"
    assert SessionKeys.LAST_DETECTION == "last_detection"
    assert SessionKeys.LAST_VIDEO == "last_video"
    assert SessionKeys.LAST_STORYBOARD == "last_storyboard"
    assert SessionKeys.YOLO_DETECTOR == "yolo_detector_singleton"


def test_session_keys_used_by_state_helpers() -> None:
    """The helpers in pages/_state.py must reference the SessionKeys constants."""
    from pages import _state

    src = Path(_state.__file__).read_text(encoding="utf-8")
    assert "SessionKeys" in src
