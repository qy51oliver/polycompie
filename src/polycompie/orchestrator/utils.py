import json
import re
from typing import Any, List, Optional, Tuple

# ```json ... ``` (or just ``` ... ```) markdown code fence wrapping JSON.
# Open-source models love wrapping output this way despite "no prose" instructions.
_FENCE_PATTERN = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```")


def parse_json(text: str) -> Tuple[Any, Optional[str]]:
    """Parse JSON from LLM text; returns (data, None) or (None, error_msg).

    Tries, in order:
      1. The whole string as JSON.
      2. Each ```json ... ``` markdown fence's contents.
      3. The widest balanced `{...}` and `[...]` substrings (whichever parses).

    Open-source models (Qwen, Llama, etc.) routinely wrap JSON in markdown
    fences with prose preamble and trailing commentary; the fence path handles
    those cleanly. The widest-balanced path is a last-resort for outputs that
    still parse despite messy formatting.
    """
    if not text or not isinstance(text, str):
        return None, "Empty or non-string input"
    s = text.strip()
    if not s:
        return None, "Empty json input"

    # 1. Try the whole string.
    try:
        return json.loads(s), None
    except json.JSONDecodeError:
        pass

    # 2. Try each ```...``` fence (prefer the longest, in case the model
    #    emitted multiple — last fence is usually the consolidated result).
    fences: List[str] = [m.group(1).strip() for m in _FENCE_PATTERN.finditer(s)]
    for inner in sorted(fences, key=len, reverse=True):
        if not inner:
            continue
        try:
            return json.loads(inner), None
        except json.JSONDecodeError:
            pass

    # 3. Last resort: widest outer braces / brackets substring. Try both
    #    `{...}` and `[...]` so we don't accidentally grab an inner array
    #    when the outer wrapper is an object.
    candidates: List[str] = []
    fb, lb = s.find("{"), s.rfind("}")
    if 0 <= fb < lb:
        candidates.append(s[fb:lb + 1])
    fs, ls = s.find("["), s.rfind("]")
    if 0 <= fs < ls:
        candidates.append(s[fs:ls + 1])
    # Longest first — the outer wrapper is wider than any inner sub-structure.
    for cand in sorted(candidates, key=len, reverse=True):
        try:
            return json.loads(cand), None
        except json.JSONDecodeError:
            pass

    return None, "invalid JSON"