"""Read original annotations without losing repeated property observations."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


class _ObjectPairs(list):
    """Distinguish a JSON object from an ordinary array before resolving its keys."""


def _sample_properties(path):
    if path[:1] == ("samples",):
        path = path[1:]
    return len(path) == 2 and isinstance(path[0], int) and path[1] == "properties"


def _unpack(value, path=()):
    if isinstance(value, _ObjectPairs):
        counts = Counter(key for key, _ in value)
        numbers = Counter()
        result = {}
        for key, item in value:
            name = key
            if counts[key] > 1:
                if not _sample_properties(path):
                    raise ValueError(f"Repeated JSON key: {key!r} at {path}")
                suffix = re.fullmatch(r"(.+)_(\d+)", key)
                if suffix is None or key in result:
                    base = suffix[1] if suffix else key
                    if suffix:
                        numbers[base] = max(numbers[base], int(suffix[2]))
                    while True:
                        numbers[base] += 1
                        name = f"{base}_{numbers[base]}"
                        if name not in counts and name not in result:
                            break
            result[name] = _unpack(item, (*path, name))
        return result
    if isinstance(value, list):
        return [_unpack(item, (*path, index)) for index, item in enumerate(value)]
    return value


def parse_gt(path: Path) -> tuple[Any, str | None]:
    """Parse original annotation values, preserving duplicate property keys.

    The historical control-character and trailing-comma cleanup is retained.
    Repeated property keys receive collision-safe numeric suffixes in memory;
    the source file and measurement values are never changed.
    """
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None, "empty file"
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
        cleaned = re.sub(r",(\s*[\]\}])", r"\1", cleaned)
        return _unpack(json.loads(cleaned, object_pairs_hook=_ObjectPairs)), None
    except (OSError, UnicodeError, ValueError, KeyError, IndexError, TypeError) as exc:
        return None, f"Annotation could not be read: {exc}"
