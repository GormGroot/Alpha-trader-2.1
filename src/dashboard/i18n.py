"""
Internationalization (i18n) module for the Alpha Trading Platform dashboard.

Loads .lan files (JSON format) from the lang/ directory.
Provides a simple t(key) translation function.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from loguru import logger

_LANG_DIR = Path(__file__).resolve().parent.parent.parent / "lang"
_cache: dict[str, dict] = {}
_current_lang: str = "da"


def get_languages_config() -> dict:
    """Load languages.json config."""
    config_path = _LANG_DIR / "languages.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load languages.json: {e}")
        return {"default": "da", "available": []}


def get_available_languages() -> list[dict]:
    """Return list of available languages as [{'code': ..., 'name': ...}, ...]."""
    cfg = get_languages_config()
    return cfg.get("available", [])


def _load_lang(code: str) -> dict:
    """Load a .lan file by language code, cache the result."""
    if code in _cache:
        return _cache[code]

    cfg = get_languages_config()
    filename = None
    for lang in cfg.get("available", []):
        if lang["code"] == code:
            filename = lang["file"]
            break

    if not filename:
        logger.warning(f"Language '{code}' not found in languages.json")
        return {}

    filepath = _LANG_DIR / filename
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        _cache[code] = data
        return data
    except Exception as e:
        logger.warning(f"Could not load {filepath}: {e}")
        return {}


def set_language(code: str) -> None:
    """Set the active language."""
    global _current_lang
    _current_lang = code


def get_language() -> str:
    """Get the current language code."""
    return _current_lang


def t(key: str, lang: str | None = None) -> str | list:
    """
    Translate a key using dot notation, e.g. t('nav.portfolio').

    Returns a string for scalar values, or a list for array values.
    Falls back to English, then returns the key itself if not found.
    """
    code = lang or _current_lang
    data = _load_lang(code)

    # Walk dot-separated path
    parts = key.split(".")
    val = data
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            val = None
            break

    if val is not None and isinstance(val, (str, list)):
        return val

    # Fallback to English
    if code != "en":
        en_data = _load_lang("en")
        val = en_data
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break
        if val is not None and isinstance(val, (str, list)):
            return val

    # Last resort: return key
    return key


def reload_languages() -> None:
    """Clear cache and reload all language files."""
    _cache.clear()
