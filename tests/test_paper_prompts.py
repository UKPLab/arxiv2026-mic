"""Pin construction templates to the three listings in the supplied manuscript."""

import hashlib

import pytest

from data_construction import prompts
from src.schema import INCONSISTENCY_TYPES


# SHA-256 of each original LaTeX lstlisting body, excluding boundary newlines.
# Preserve placeholder spelling while rendering Python's escaped JSON braces.
@pytest.mark.parametrize("name,expected", [
    ("FILTER_PROMPT", "4336e168458797137ca78adf40b088fafb59a286343a0b2f7126bc6d4eaed98e"),
    ("STRICT_PROMPT", "c039eca7a9ad7bad24aae0e7c54176fbb7bd964759af66726518941ff8fb8ea6"),
    ("EDIT_PROMPT_TEMPLATE", "99f556c93bb74bae972013d3dfafae8d2e524586f0f65a3c951964cb3058d08d"),
])
def test_construction_prompt_matches_manuscript_listing(name, expected):
    fields = {key: "{" + key + "}" for key in (
        "caption", "headline", "location", "time", "keywords", "visual_description",
        "edit_type", "evidence", "inconsistency_def", "region_hint",
    )}
    fields["type_descs"] = {kind: "{type_descs[" + kind + "]}" for kind in INCONSISTENCY_TYPES}
    rendered = getattr(prompts, name).format(**fields)
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == expected
