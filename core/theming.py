"""Theme helpers: produce the CSS string injected into Streamlit.

The function `get_theme_css(theme)` is pure — it takes a theme name
(`"light"` or `"dark"`) and returns a `<style>` block tailored to that
theme. Every colour, radius, font size lives in one place so the
visual identity can be tweaked in a single file.

The CSS uses tokens that are widely understood by browsers and the
Streamlit DOM:

- `var(--primary)` — accent colour (buttons, links, active tabs)
- `var(--bg)` — page background
- `var(--surface)` — card background
- `var(--text)` — body text
- `var(--muted)` — captions / helper text
- `var(--border)` — card / input border
- `var(--success)`, `var(--danger)`, `var(--warning)` — status colours
- `var(--focus-ring)` — accessibility focus outline

Every theme additionally embeds an **accessibility chunk**
(`_A11Y_CSS`) that provides:

- A keyboard-only skip link.
- High-visibility focus rings on interactive elements.
- A `prefers-reduced-motion` guard that disables transitions.
- WCAG-friendly colour-contrast tokens (via the CSS variables above).
"""
from __future__ import annotations

_A11Y_CSS: str = """
/* --- Accessibility additions ----------------------------------------- */
/* Skip-to-content link, visible only when focused. */
.skip-link {
  position: absolute;
  left: -9999px;
  top: 0.5rem;
  background: var(--primary);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: var(--radius);
  z-index: 9999;
  font-weight: 600;
}
.skip-link:focus { left: 0.5rem; outline: 2px solid white; }

/* High-visibility focus rings on every interactive surface. */
:focus-visible {
  outline: none;
  box-shadow: var(--focus-ring);
  border-radius: var(--radius);
}

/* Respect users who prefer reduced motion. */
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}

/* Screen-reader-only utility class. */
.sr-only {
  position: absolute !important;
  width: 1px; height: 1px;
  padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}
"""

_VALID_THEMES: frozenset[str] = frozenset({"light", "dark", "deep_blue"})


class InvalidThemeError(ValueError):
    """Raised when an unknown theme name is requested."""


_THEMES: dict[str, str] = {
    "light": """
:root {
  --primary: #1e3a8a;
  --primary-hover: #1e40af;
  --accent: #2563eb;
  --bg: #f8fafc;
  --surface: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
  --success: #16a34a;
  --warning: #d97706;
  --danger: #b91c1c;
  --shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
  --radius: 6px;
  --focus-ring: 0 0 0 3px rgba(37, 99, 235, 0.35);
  --font-sans: "Inter", "Roboto", system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
  font-size: 15px;
  line-height: 1.55;
}
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
h1, h2, h3, h4 { color: var(--text) !important; font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 1.6rem; margin-top: 0; }
h2 { font-size: 1.25rem; }
h3 { font-size: 1.05rem; }
.stMarkdown, .stText, p, span, label { color: var(--text); }
.stButton > button {
  background: var(--primary);
  color: #ffffff;
  border: 0;
  border-radius: var(--radius);
  padding: 0.55rem 1.1rem;
  font-weight: 500;
  box-shadow: var(--shadow);
  transition: background 120ms ease;
}
.stButton > button:hover { background: var(--primary-hover); }
.stButton > button:focus-visible { outline: none; box-shadow: var(--focus-ring); }
[data-testid="stMetricValue"] { color: var(--primary); font-weight: 600; }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.85rem; }
[data-testid="stMetricDelta"] { color: var(--muted); }
[data-testid="stExpander"], div[data-baseweb="card"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.stProgress > div > div > div { background: var(--primary); }
.stAlert[data-baseweb="notification"] { border-radius: var(--radius); border: 1px solid var(--border); }
code, pre { font-family: var(--font-mono) !important; }
.brand-footer {
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.78rem;
  color: var(--muted);
  line-height: 1.5;
}
.brand-header {
  padding: 0.85rem 0.25rem 1rem 0.25rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.65rem;
}
.brand-mark {
  width: 30px; height: 30px;
  background: var(--primary);
  color: #ffffff;
  font-weight: 700;
  font-size: 0.85rem;
  border-radius: 4px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  letter-spacing: 0.02em;
}
.brand-name { font-weight: 600; font-size: 0.95rem; color: var(--text); }
.brand-tag  { display: block; font-size: 0.72rem; color: var(--muted); margin-top: 0.05rem; }
.cv-empty {
  border: 1px dashed var(--border);
  background: var(--surface);
  border-radius: var(--radius);
  padding: 1.5rem 1.25rem;
  margin: 0.5rem 0 1rem 0;
}
.cv-empty-title { font-weight: 600; margin: 0 0 0.25rem 0; color: var(--text); }
.cv-empty-body  { color: var(--muted); margin: 0 0 0.75rem 0; font-size: 0.92rem; }
.cv-pill {
  display: inline-block;
  padding: 0.1rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.cv-pill--ok       { color: var(--success); border-color: var(--success); background: rgba(22,163,74,0.06); }
.cv-pill--warn     { color: var(--warning); border-color: var(--warning); background: rgba(217,119,6,0.06); }
.cv-pill--danger   { color: var(--danger);  border-color: var(--danger);  background: rgba(185,28,28,0.06); }
""",
    "dark": """
:root {
  --primary: #60a5fa;
  --primary-hover: #93c5fd;
  --accent: #60a5fa;
  --bg: #0f172a;
  --surface: #111827;
  --text: #e5e7eb;
  --muted: #94a3b8;
  --border: #1f2937;
  --success: #34d399;
  --warning: #fbbf24;
  --danger: #f87171;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  --radius: 6px;
  --focus-ring: 0 0 0 3px rgba(96, 165, 250, 0.45);
  --font-sans: "Inter", "Roboto", system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
  font-size: 15px;
  line-height: 1.55;
}
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
h1, h2, h3, h4 { color: var(--text) !important; font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 1.6rem; margin-top: 0; }
h2 { font-size: 1.25rem; }
h3 { font-size: 1.05rem; }
.stMarkdown, .stText, p, span, label { color: var(--text); }
.stButton > button {
  background: var(--primary);
  color: #0b1220;
  border: 0;
  border-radius: var(--radius);
  padding: 0.55rem 1.1rem;
  font-weight: 500;
  box-shadow: var(--shadow);
  transition: background 120ms ease;
}
.stButton > button:hover { background: var(--primary-hover); }
.stButton > button:focus-visible { outline: none; box-shadow: var(--focus-ring); }
[data-testid="stMetricValue"] { color: var(--primary); font-weight: 600; }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.85rem; }
[data-testid="stMetricDelta"] { color: var(--muted); }
[data-testid="stExpander"], div[data-baseweb="card"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.stProgress > div > div > div { background: var(--primary); }
.stAlert[data-baseweb="notification"] { border-radius: var(--radius); border: 1px solid var(--border); }
code, pre { font-family: var(--font-mono) !important; }
.brand-footer {
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.78rem;
  color: var(--muted);
  line-height: 1.5;
}
.brand-header {
  padding: 0.85rem 0.25rem 1rem 0.25rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.65rem;
}
.brand-mark {
  width: 30px; height: 30px;
  background: var(--primary);
  color: #0b1220;
  font-weight: 700;
  font-size: 0.85rem;
  border-radius: 4px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  letter-spacing: 0.02em;
}
.brand-name { font-weight: 600; font-size: 0.95rem; color: var(--text); }
.brand-tag  { display: block; font-size: 0.72rem; color: var(--muted); margin-top: 0.05rem; }
.cv-empty {
  border: 1px dashed var(--border);
  background: var(--surface);
  border-radius: var(--radius);
  padding: 1.5rem 1.25rem;
  margin: 0.5rem 0 1rem 0;
}
.cv-empty-title { font-weight: 600; margin: 0 0 0.25rem 0; color: var(--text); }
.cv-empty-body  { color: var(--muted); margin: 0 0 0.75rem 0; font-size: 0.92rem; }
.cv-pill {
  display: inline-block;
  padding: 0.1rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.cv-pill--ok       { color: var(--success); border-color: var(--success); background: rgba(52,211,153,0.08); }
.cv-pill--warn     { color: var(--warning); border-color: var(--warning); background: rgba(251,191,36,0.08); }
.cv-pill--danger   { color: var(--danger);  border-color: var(--danger);  background: rgba(248,113,113,0.08); }
""",
    "deep_blue": """
:root {
  --primary: #00d4ff;
  --primary-hover: #00b8e6;
  --accent: #00d4ff;
  --bg: #0b1120;
  --surface: #171f33;
  --text: #f8fafc;
  --muted: #94a3b8;
  --border: #2d3a5a;
  --success: #4ade80;
  --warning: #fbbf24;
  --danger: #f87171;
  --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
  --radius: 8px;
  --focus-ring: 0 0 0 3px rgba(0, 212, 255, 0.4);
  --font-sans: "Inter", "Roboto", system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
  font-size: 15px;
  line-height: 1.6;
}
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
h1, h2, h3, h4 { color: #ffffff !important; font-weight: 700; letter-spacing: -0.02em; }
h1 { font-size: 1.8rem; margin-top: 0; }
h2 { font-size: 1.4rem; }
h3 { font-size: 1.1rem; }
.stMarkdown, .stText, p, span, label { color: var(--text); }
.stButton > button {
  background: var(--primary);
  color: #0b1120;
  border: 0;
  border-radius: var(--radius);
  padding: 0.6rem 1.2rem;
  font-weight: 600;
  box-shadow: var(--shadow);
  transition: all 150ms ease;
}
.stButton > button:hover { background: var(--primary-hover); transform: translateY(-1px); }
.stButton > button:focus-visible { outline: none; box-shadow: var(--focus-ring); }
[data-testid="stMetricValue"] { color: var(--primary); font-weight: 700; }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.85rem; }
[data-testid="stMetricDelta"] { color: var(--muted); }
[data-testid="stExpander"], div[data-baseweb="card"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
}
.stProgress > div > div > div { background: var(--primary); }
.stAlert[data-baseweb="notification"] { border-radius: var(--radius); border: 1px solid var(--border); }
code, pre { font-family: var(--font-mono) !important; color: var(--primary) !important; }
.brand-footer {
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.78rem;
  color: var(--muted);
  line-height: 1.5;
}
.brand-header {
  padding: 0.85rem 0.25rem 1rem 0.25rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.65rem;
}
.brand-mark {
  width: 30px; height: 30px;
  background: var(--primary);
  color: #0b1120;
  font-weight: 700;
  font-size: 0.85rem;
  border-radius: 4px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  letter-spacing: 0.02em;
}
.brand-name { font-weight: 700; font-size: 1rem; color: #ffffff; }
.brand-tag  { display: block; font-size: 0.72rem; color: var(--muted); margin-top: 0.05rem; }
.cv-empty {
  border: 1px dashed var(--border);
  background: var(--surface);
  border-radius: var(--radius);
  padding: 1.5rem 1.25rem;
  margin: 0.5rem 0 1rem 0;
}
.cv-empty-title { font-weight: 700; margin: 0 0 0.25rem 0; color: #ffffff; }
.cv-empty-body  { color: var(--muted); margin: 0 0 0.75rem 0; font-size: 0.92rem; }
.cv-pill {
  display: inline-block;
  padding: 0.1rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.cv-pill--ok       { color: var(--success); border-color: var(--success); background: rgba(74, 222, 128, 0.1); }
.cv-pill--warn     { color: var(--warning); border-color: var(--warning); background: rgba(251, 191, 36, 0.1); }
.cv-pill--danger   { color: var(--danger);  border-color: var(--danger);  background: rgba(248, 113, 113, 0.1); }
""",
}


def available_themes() -> tuple[str, ...]:
    """Return the supported theme names."""
    return tuple(sorted(_VALID_THEMES))


def get_theme_css(theme: str = "light") -> str:
    """Return the CSS block for the requested theme.

    The returned string combines the theme-specific tokens with the
    shared accessibility chunk (`_A11Y_CSS`) so every theme ships
    keyboard / focus / motion support out of the box.

    Raises `InvalidThemeError` for unknown names so misconfigurations
    surface immediately instead of silently rendering the default.
    """
    if theme not in _VALID_THEMES:
        raise InvalidThemeError(
            f"unknown theme {theme!r}; choose from {sorted(_VALID_THEMES)}"
        )
    return _THEMES[theme] + _A11Y_CSS


def render_theme_markdown(theme: str = "light") -> str:
    """Return the markdown-ready CSS wrapped in a `<style>` block."""
    css = get_theme_css(theme)
    return f"<style>{css}</style>"