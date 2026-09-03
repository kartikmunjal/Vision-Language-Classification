from __future__ import annotations

import re

from .schema import TASKS

_PATTERNS = {
    "multiple_subjects": (
        r"\b(?:two|three|four|five|several|many|multiple|group|crowd|people|men|women|children)\b",
    ),
    "outdoor": (
        r"\b(?:outdoor|outside|street|road|park|forest|beach|field|mountain|ocean|river|lake|sky)\b",
    ),
    "human_present": (
        r"\b(?:person|people|man|woman|boy|girl|child|children|worker|player|rider|skier|surfer)\b",
    ),
    "animal_present": (
        r"\b(?:animal|dog|cat|horse|bird|cow|sheep|elephant|bear|zebra|giraffe|pet)\b",
    ),
    "dynamic_scene": (
        r"\b(?:running|walking|riding|driving|flying|jumping|dancing|swimming|moving|playing|throwing|skiing|surfing)\b",
    ),
    "night": (r"\b(?:night|nighttime|dark sky|moonlit|after dark)\b",),
}


def label_caption(caption: str) -> dict[str, dict[str, float | int]]:
    """Return deterministic binary labels and auditable match confidence.

    A negative means no positive pattern was stated, not proof that the visual
    attribute is absent. This asymmetry is measured against human labels.
    """
    text = caption.casefold()
    result = {}
    for task in TASKS:
        hits = [pattern for pattern in _PATTERNS[task] if re.search(pattern, text)]
        result[task] = {"label": int(bool(hits)), "confidence": 1.0 if hits else 0.55}
    return result


def patterns() -> dict[str, tuple[str, ...]]:
    return dict(_PATTERNS)
