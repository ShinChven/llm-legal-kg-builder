"""
Utility helpers for normalizing legislation titles so we can make consistent
string comparisons across different extraction pipelines.
"""

from __future__ import annotations


TITLE_NORMALIZATION_TABLE = str.maketrans(
    {
        ord("’"): "'",
        ord("‘"): "'",
        ord("“"): '"',
        ord("”"): '"',
        ord("–"): "-",
        ord("—"): "-",
        ord("‑"): "-",
        ord("‒"): "-",
        ord("−"): "-",
    }
)


def normalize_title(title: str | None) -> str:
    """
    Replaces curly quotes and dash variants with their ASCII equivalents.

    Args:
        title: Original title that may contain smart quotes or unicode dashes.

    Returns:
        A normalized string. Empty string when `title` is falsy.
    """
    if not title:
        return ""
    return title.translate(TITLE_NORMALIZATION_TABLE)


def normalize_text_for_search(text: str | None) -> str:
    """
    Lowercase version of `normalize_title` for tolerant substring matching.

    Args:
        text: Text to normalize and lowercase.
    """
    if not text:
        return ""
    return normalize_title(text).lower()


def normalize_for_comparison(text: str | None) -> str:
    """
    Normalizes and lowercases text for equality comparisons (ignores leading/trailing whitespace).
    """
    if not text:
        return ""
    return normalize_title(text).strip().lower()
