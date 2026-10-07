"""Security helpers: filename sanitisation, path traversal protection,
and input validation.

Every helper here is pure — no I/O beyond a single `Path.resolve()`
and no Streamlit imports. The defensive pattern is:

  base = Path("/safe/root").resolve()
  target = safe_join(base, user_supplied_name)
  # target is guaranteed to live under `base`, or
  # `PathTraversalError` is raised.

`safe_join()` is the only function in the project that should be used
whenever a user-controlled filename is concatenated to a server-side
directory.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Disallow everything except A–Z a–z 0–9 dot dash underscore. The
# underscore/dot/dash allow common file extensions; spaces and
# unicode separators are normalised to underscore.
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_LEADING_DOTS_RE = re.compile(r"^\.+")


class PathTraversalError(ValueError):
    """Raised when a user-supplied path resolves outside of `base`."""


class UnsafeFilenameError(ValueError):
    """Raised when a filename contains forbidden characters."""


def safe_filename(name: str, max_length: int = 120) -> str:
    """Return a filesystem-safe version of `name`.

    - Strips the path component (everything before the final separator).
    - Replaces every run of unsafe characters with a single underscore.
    - Trims leading dots (defends against `..hidden` style names).
    - Truncates to `max_length` characters.

    Raises `UnsafeFilenameError` if the resulting name is empty.
    """
    if name is None:
        raise UnsafeFilenameError("filename is None")

    # Take only the basename, in case someone passes an absolute or
    # relative path embedded in a filename upload.
    name = os.path.basename(name.replace("\\", "/"))
    if not name:
        raise UnsafeFilenameError("filename is empty")

    cleaned = _FILENAME_SAFE_RE.sub("_", name)
    cleaned = _LEADING_DOTS_RE.sub("", cleaned)
    cleaned = cleaned.strip("_-")
    if not cleaned:
        raise UnsafeFilenameError(f"filename reduced to empty: {name!r}")
    if len(cleaned) > max_length:
        # Keep extension when truncating.
        stem, dot, ext = cleaned.rpartition(".")
        if dot and ext and len(ext) < 8:
            cleaned = stem[: max_length - len(ext) - 1] + "." + ext
        else:
            cleaned = cleaned[:max_length]
    return cleaned


def is_allowed_extension(filename: str, allowed: set[str] | frozenset[str]) -> bool:
    """Return True iff the extension (lowercased) is in `allowed`."""
    if not filename or not allowed:
        return False
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in {a.lower() for a in allowed}


def safe_join(base: Path | str, *parts: str | os.PathLike) -> Path:
    """Join `parts` under `base` and verify the result is inside `base`.

    `base` is resolved to an absolute path first. The final path is
    also resolved. If the resolved final path is not under `base`,
    `PathTraversalError` is raised.

    Use this for every user-supplied file write — never `os.path.join`.
    """
    base_resolved = Path(base).resolve()
    if not parts:
        return base_resolved

    candidate = base_resolved.joinpath(*parts)
    # resolve() follows symlinks; if the target doesn't exist yet,
    # `strict=False` (default) handles that gracefully.
    final = candidate.resolve(strict=False)

    try:
        final.relative_to(base_resolved)
    except ValueError as exc:
        raise PathTraversalError(
            f"refusing to write outside base: {final} not under {base_resolved}"
        ) from exc

    return final


def validate_text_input(
    text: str | None,
    *,
    field: str,
    max_length: int = 1000,
    min_length: int = 0,
    allow_newlines: bool = True,
) -> str:
    """Validate free-text user input.

    - Rejects non-string / None.
    - Rejects strings shorter than `min_length` or longer than `max_length`.
    - Strips trailing whitespace.
    - Optionally strips embedded newlines (single-line fields).

    Raises `ValueError` with a user-readable message on failure.
    """
    if text is None:
        raise ValueError(f"{field}: value is required")
    if not isinstance(text, str):
        raise ValueError(f"{field}: must be a string")
    if len(text) < min_length:
        raise ValueError(f"{field}: must be at least {min_length} characters")
    if len(text) > max_length:
        raise ValueError(f"{field}: must be at most {max_length} characters")
    cleaned = text.strip()
    if not allow_newlines:
        cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    return cleaned


def validate_email(email: str) -> str:
    """Return a normalised email or raise `ValueError`.

    A pragmatic check — full RFC compliance isn't needed for a contact
    form. We require local@domain with at least one dot in the domain.
    """
    email = email.strip().lower()
    if "@" not in email:
        raise ValueError("email: missing '@'")
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain:
        raise ValueError("email: malformed")
    if len(email) > 254:
        raise ValueError("email: too long")
    return email