"""Parse the four final MIC answer fields and optional teacher reasoning.

Final tags must occur exactly once. Thinking may be explicit or use Qwen's
native closing tag. Missing, duplicate, or inconsistent final fields fail
validation, while extracted values remain available for inspection.
"""

import re

from ..schema import INCONSISTENCY_TYPES

FINAL_TAGS = ("verdict", "type", "visual", "explanation")
NONE_MARKERS = {"none", "null", "n/a", "na", ""}


def _normalize_free_text(raw):
    if raw is None or raw.strip().lower() in NONE_MARKERS:
        return None
    return raw.strip()


def parse(raw_text: str) -> dict:
    errors = []
    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.S | re.I)
    closes = list(re.finditer(r"</think>", raw_text, re.I))
    if closes:
        final = raw_text[closes[-1].end():]
        think = think_match.group(1).strip() if think_match else raw_text[:closes[0].start()].strip()
    else:
        final, think = raw_text, None
    fields = {}
    remainder = final
    for tag in FINAL_TAGS:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, final, re.S | re.I)
        opens = len(re.findall(rf"<{tag}>", final, re.I))
        closes = len(re.findall(rf"</{tag}>", final, re.I))
        if len(matches) != 1 or opens != 1 or closes != 1:
            errors.append(f"missing_or_duplicate_{tag}")
        fields[tag] = _normalize_free_text(matches[0]) if len(matches) == 1 else None
        remainder = re.sub(pattern, "", remainder, flags=re.S | re.I)
    if remainder.strip():
        errors.append("text_outside_final_tags")
    verdict = fields["verdict"].upper() if fields["verdict"] else None
    if verdict not in {"CONSISTENT", "INCONSISTENT"}:
        verdict = None
        errors.append("invalid_verdict")
    typ = fields["type"].lower() if fields["type"] else None
    if typ is not None and typ not in INCONSISTENCY_TYPES:
        typ = None
        errors.append("invalid_type")
    if verdict == "INCONSISTENT":
        for tag, value in (("type", typ), ("visual", fields["visual"]), ("explanation", fields["explanation"])):
            if value is None:
                errors.append(f"inconsistent_but_no_{tag}")
    elif verdict == "CONSISTENT":
        for tag in ("type", "visual", "explanation"):
            if fields[tag] is not None:
                errors.append(f"consistent_but_has_{tag}")
    return {"verdict": verdict, "type": typ, "visual": fields["visual"],
            "explanation": fields["explanation"], "think": _normalize_free_text(think),
            "parse_ok": not errors, "parse_errors": errors}
