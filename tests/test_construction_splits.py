"""Regression checks for deterministic paired entity/temporal splitting."""

from collections import Counter
from copy import deepcopy
import json
import random

import pytest

from data_construction.dataset import build_splits, split_main
from src.records import normalize_record


@pytest.fixture
def annotations():
    rows = []
    groups = [
        ("seen", "flag", [2008, 2010, 2011, 2012, 2015, 2016, 2019, 2020, 2021]),
        ("only_a", "flag", [2019]),
        ("only_b", "flag", [2019, 2020]),
        ("mixed_ood", "signage", [2011, 2019, 2020, 2021]),
        ("mixed_id", "signage", [2010, 2011, 2012, 2015, 2016, 2019, 2020]),
    ]
    for entity, edit_type, years in groups:
        for index, year in enumerate(years):
            stem = f"{entity}{index}"
            for image_type in ("original", "edited"):
                rows.append(normalize_record({
                    "_id": stem, "claim": stem, "image_type": image_type,
                    "local_image": f"{stem}_{image_type}.png",
                    "year": year if image_type == "edited" else 1,
                    "edit_type": edit_type if image_type == "edited" else None,
                    "gt_entity_canonical": entity, "gt_entity_fine": "A visual",
                    "gt_why_contradicts": "A reason",
                }))
    return rows


def test_split_assignment_preserves_ranking_order_and_discard_reasons(annotations):
    result = build_splits(annotations, ood_target=1)
    expected = {
        "train": ["mixed_id2", "mixed_id1", "mixed_id4", "seen0", "seen3", "seen2", "seen5"],
        "val": ["mixed_id0", "mixed_id3", "seen1", "seen4"],
        "test_id_edit": ["mixed_id5", "mixed_id6", "seen8", "seen6"],
        "test_ood_edit": ["mixed_ood1", "mixed_ood2", "mixed_ood3", "only_b0", "only_b1"],
    }
    assert {name: [row["_id"] for row in result[name][1::2]] for name in expected} == expected
    # Equal recent/total ratios prefer entities with more recent rows.
    assert result["ood_entities"] == {"flag": ["only_b"], "signage": ["mixed_ood"]}
    assert result["discarded"] == [
        {"stem": "mixed_ood0", "edit_type": "signage", "entity": "mixed_ood",
         "year": 2011, "reason": "ood_pre_cut"},
        {"stem": "only_a0", "edit_type": "flag", "entity": "only_a",
         "year": 2019, "reason": "all_recent_no_train_anchor"},
        {"stem": "seen7", "edit_type": "flag", "entity": "seen",
         "year": 2020, "reason": "id_recent_surplus"},
    ]
    report = result["report"]
    assert "| flag | 4 | 2 | 2 | 2 | 2 |\n| signage | 3 | 2 | 2 | 3 | 1 |" in report
    assert "| **TOTAL** | 7 | 4 | 4 | 5 | 3 |" in report
    assert "| train | 1 | 4 | 2 | 0 | 7 |" in report
    assert "- ID test entities absent from training: 0 (expected: 0)" in report
    assert "- OOD test entities present in training: 0 (expected: 0)" in report


@pytest.mark.parametrize("year_cut", [2014, 2018, 2020])
def test_splits_are_pure_deterministic_and_keep_pairs_and_entities_together(annotations, year_cut):
    before = deepcopy(annotations)
    rng_state = random.getstate()
    result = build_splits(annotations, year_cut=year_cut, ood_target=1)
    assert result == build_splits(list(reversed(annotations)), year_cut=year_cut, ood_target=1)
    assert annotations == before
    assert random.getstate() == rng_state
    owners = {}
    entities = {}
    counts = {}
    for name in ("train", "val", "test_id_edit", "test_ood_edit"):
        rows = result[name]
        entities[name] = set()
        counts[name] = Counter()
        for original, edited in zip(rows[::2], rows[1::2]):
            article_id = edited["_id"]
            assert article_id not in owners
            owners[article_id] = name
            assert original["_id"] == article_id
            assert (original["image_type"], edited["image_type"]) == ("original", "edited")
            assert original["year"] == edited["year"]
            assert original["edit_type"] == edited["edit_type"]
            assert (edited["year"] < year_cut) == (name in {"train", "val"})
            entities[name].add(edited["gt_entity_canonical"])
            counts[name][edited["edit_type"]] += 1
    assert entities["test_id_edit"] <= entities["train"]
    assert not entities["test_ood_edit"] & entities["train"]
    assert all(count <= counts["test_ood_edit"][typ] for typ, count in counts["test_id_edit"].items())
    assert set(owners).isdisjoint(row["stem"] for row in result["discarded"])
    assert len(owners) + len(result["discarded"]) == len(annotations) // 2


def test_cli_aliases_write_identical_artifacts(annotations, tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(annotations))
    outputs = []
    for separator in ("_", "-"):
        output = tmp_path / separator
        split_main(["--annotations", str(source), "--output-dir", str(output),
              f"--year{separator}cut", "2018", f"--ood{separator}target", "1",
              f"--val{separator}frac", "0.1"])
        outputs.append({path.name: path.read_bytes() for path in output.iterdir()})
    assert outputs[0] == outputs[1]
    assert set(outputs[0]) == {"train.json", "val.json", "test_id_edit.json", "test_ood_edit.json",
                               "discarded.json", "ood_entities.json", "split_report.md"}
    assert b"- Source: `source.json`" in outputs[0]["split_report.md"]


@pytest.mark.parametrize("options", [{"val_frac": -0.1}, {"val_frac": 1}, {"ood_target": 0}])
def test_split_options_are_validated(annotations, options):
    with pytest.raises(ValueError, match="val_frac must be"):
        build_splits(annotations, **options)


@pytest.mark.parametrize("field,value", [("year", "2019"), ("gt_entity_canonical", None)])
def test_split_metadata_is_required(annotations, field, value):
    annotations[1][field] = value
    with pytest.raises(ValueError, match="entity splitting requires"):
        build_splits(annotations)


def test_cross_type_entities_and_incomplete_pairs_are_rejected(annotations):
    with pytest.raises(ValueError, match="Incomplete"):
        build_splits(annotations[1:])
    annotations[1]["gt_entity_canonical"] = "mixed_id"
    with pytest.raises(ValueError, match="multiple edit types"):
        build_splits(annotations)


def test_failed_split_does_not_write_partial_artifacts(annotations, tmp_path):
    recent_only = [row for row in annotations if row["_id"].startswith("only_")]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(recent_only))
    output = tmp_path / "splits"
    with pytest.raises(ValueError, match="No training pairs remain"):
        split_main(["--annotations", str(source), "--output-dir", str(output), "--ood-target", "1"])
    assert not output.exists()
