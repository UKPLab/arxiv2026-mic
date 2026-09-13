"""Release regressions for corrupt artifacts, review provenance, and paid-call resumption."""

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from data_construction import common, dataset, images
from src.records import normalize_record
from src.training.prepare import build_target


@pytest.fixture
def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(stream, format="PNG")
    return stream.getvalue()


def reviewed_pair(root, claim_id="article", entity="flag", year=2010):
    rows = []
    for image_type in ("original", "edited"):
        path = root / f"{claim_id}_{image_type}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        rows.append(normalize_record({
            "_id": claim_id, "image_type": image_type, "local_image": path.name,
            "claim": "An official ceremony.", "edit_type": "flag", "year": year,
            "gt_entity_fine": "The replacement flag", "gt_entity_canonical": entity,
            "gt_why_contradicts": "The replacement flag contradicts the ceremony's location.",
        }))
    return rows


def test_invalid_download_and_cache_do_not_become_successes(tmp_path, monkeypatch, image_bytes):
    record = {"_id": "a", "image_url": "https://example.test/image-articleLarge.jpg?token=opaque"}
    corrupt = b"<html>Access denied</html>" * 100
    cached = tmp_path / images.url_to_filename(record["image_url"])
    cached.write_bytes(corrupt)
    get = Mock(side_effect=[SimpleNamespace(status_code=200, content=corrupt),
                            SimpleNamespace(status_code=200, content=image_bytes)])
    monkeypatch.setattr(requests, "get", get)
    _, filename, status = images.download_one(record, tmp_path)
    assert status == "ok_fallback" and filename == cached.name
    assert cached.read_bytes() == image_bytes
    assert get.call_args_list[-1].args[0] == record["image_url"]
    assert common.image_to_base64_uri(cached).startswith("data:image/png;base64,")
    assert sorted(tmp_path.iterdir()) == [cached]


def test_invalid_generated_bytes_do_not_overwrite_previous_image(tmp_path, image_bytes):
    original = tmp_path / "original.png"
    original.write_bytes(image_bytes)
    content = [image_bytes]
    client = SimpleNamespace(images=SimpleNamespace(edit=lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(content[0]).decode())],
    )))
    item = {"_id": "a", "local_image": original.name, "edit_result": {"editing_prompt": "Replace the flag."}}
    _, first = images.edit_one(client, item, tmp_path, tmp_path / "edited")
    output = tmp_path / "edited" / first["output_image"]
    content[0] = b"invalid image response"
    _, failed = images.edit_one(client, item, tmp_path, tmp_path / "edited")
    assert "invalid image bytes" in failed["error"]
    assert output.read_bytes() == image_bytes


def test_regeneration_keeps_reviewed_image_and_requires_review_of_new_content(tmp_path, image_bytes):
    source = tmp_path / "original.png"
    source.write_bytes(image_bytes)
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(stream, format="PNG")
    variants = iter([image_bytes, stream.getvalue()])
    client = SimpleNamespace(images=SimpleNamespace(edit=lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(next(variants)).decode())],
    )))
    item = {"_id": "a", "caption": "An official ceremony.", "local_image": source.name,
            "edit_type": "flag", "edit_result": {"editing_prompt": "Replace the flag."}}
    _, previous = images.edit_one(client, item, tmp_path, tmp_path)
    reviews = dataset.review_template([dict(item, edit_status=previous)])
    reviews[0].update(accepted=True, year=2010, gt_entity_canonical="flag", gt_entity_fine="Flag",
                      gt_why_contradicts="Wrong flag for the location.")
    _, regenerated = images.edit_one(client, item, tmp_path, tmp_path)
    assert previous["output_image"] != regenerated["output_image"]
    assert (tmp_path / previous["output_image"]).read_bytes() == image_bytes
    with pytest.raises(ValueError, match="review images differ"):
        dataset.assemble([dict(item, edit_status=regenerated)], reviews, tmp_path, ".", ".")


@pytest.mark.parametrize("result_key,result", [
    ("filter_result", {"suitable": "false", "feasible_types": ["flag"]}),
    ("filter_result", {"suitable": True, "feasible_types": []}),
    ("round2_result", {"quality": "high", "feasible_types": {"flag": {"visible": "false"}}}),
    ("round2_result", {"quality": "high", "feasible_types": ["flag"]}),
    ("edit_result", {"editing_prompt": "  "}),
    ("edit_result", {}),
])
def test_malformed_model_fields_are_not_completed_results(result_key, result):
    assert not images.stage_completed({result_key: result}, result_key)
    with pytest.raises(ValueError):
        images.validate_stage_result(result, result_key)


def test_text_false_is_never_a_visible_edit_target():
    assert images.get_visible_types({"round2_result": {
        "feasible_types": {"flag": {"visible": "false"}},
    }}) == []


@pytest.mark.parametrize("size,accepted", [
    ((1023, 1024), False),
    ((1024, 1023), False),
    ((1024, 1024), True),
])
def test_screen_requires_both_paper_image_dimensions(tmp_path, monkeypatch, size, accepted):
    source = tmp_path / "image.png"
    Image.new("RGB", size, "white").save(source)
    response = {"suitable": False, "feasible_types": []}
    chat = Mock(return_value=response)
    monkeypatch.setattr(images, "chat_json", chat)
    record = {"_id": "a", "caption": "A ceremony.", "local_image": source.name}

    claim_id, result = images.filter_one(object(), record, tmp_path)

    assert claim_id == record["_id"]
    if accepted:
        assert result == response
        chat.assert_called_once()
    else:
        assert result == {"error": f"low_resolution_{size[0]}x{size[1]}px"}
        chat.assert_not_called()


def test_malformed_screen_is_saved_as_failure_then_retried(tmp_path, monkeypatch, image_bytes):
    source = tmp_path / "image.png"
    source.write_bytes(image_bytes)
    metadata, output = tmp_path / "metadata.json", tmp_path / "screen.json"
    common.write_json(metadata, [{"_id": "a", "caption": "A ceremony.", "local_image": source.name}])
    responses = iter([{"suitable": "false", "feasible_types": ["flag"]},
                      {"suitable": False, "feasible_types": []}])
    calls = []

    def chat(*args, **kwargs):
        calls.append(True)
        return next(responses)

    monkeypatch.setattr(common, "create_client", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(images, "chat_json", chat)
    args = ["--metadata", str(metadata), "--output", str(output), "--image-dir", str(tmp_path),
            "--min-width", "0", "--min-height", "0"]
    images.filter_main(args)
    assert "error" in common.read_json(output)[0]["filter_result"]
    images.filter_main(args + ["--resume"])
    assert common.read_json(output)[0]["filter_result"] == {"suitable": False, "feasible_types": []}
    images.filter_main(args + ["--resume"])
    assert len(calls) == 2


def test_resume_rejects_records_removed_from_current_source_before_api_setup(tmp_path, monkeypatch):
    output = tmp_path / "screen.json"
    common.write_json(output, [{"_id": "removed", "result": {"ok": True}}])
    monkeypatch.setattr(common, "create_client", lambda: pytest.fail("Changed source must be checked before paid calls"))
    with pytest.raises(ValueError, match="outside the current source"):
        common.run_api_batch([{"_id": "new"}], lambda row, client: row, output, result_key="result", resume=True)
    assert common.read_json(output)[0]["_id"] == "removed"


@pytest.mark.parametrize("change", ["stale_image", "duplicate", "boolean_year", "blank_entity"])
def test_review_requires_current_images_and_valid_split_metadata(tmp_path, change):
    original, edited = reviewed_pair(tmp_path)
    results = [{"_id": original["_id"], "caption": original["claim"], "local_image": original["local_image"],
                "edit_type": "flag", "edit_status": {"output_image": edited["local_image"]}}]
    reviews = dataset.review_template(results)
    reviews[0].update(accepted=True, year=2010, gt_entity_canonical="flag",
                      gt_entity_fine=edited["gt_entity_fine"], gt_why_contradicts=edited["gt_why_contradicts"])
    if change == "stale_image":
        reviews[0]["edited_image"] = "previous_generation.png"
    elif change == "duplicate":
        reviews.append(dict(reviews[0], accepted=False))
    elif change == "boolean_year":
        reviews[0]["year"] = True
    else:
        reviews[0]["gt_entity_canonical"] = "  "
    with pytest.raises(ValueError):
        dataset.assemble(results, reviews, tmp_path, ".", ".")


def test_teacher_resume_retains_successes_and_retries_rejected_response(tmp_path, monkeypatch):
    from src.backends import api_backend

    rows = reviewed_pair(tmp_path)
    source, output = tmp_path / "source.json", tmp_path / "teacher.json"
    common.write_json(source, rows)
    calls, closed = [], []
    targets = {row["local_image"]: build_target(row) for row in rows}
    fail = [True]

    class Backend:
        def __init__(self, *args):
            pass

        def load(self):
            pass

        def unload(self):
            closed.append(True)

        def run(self, path, prompt):
            name = Path(path).name
            calls.append(name)
            if fail[0] and name == rows[1]["local_image"]:
                return targets[rows[0]["local_image"]]
            return targets[name]

    monkeypatch.setattr(api_backend, "APIBackend", Backend)
    args = ["--input", str(source), "--output", str(output), "--data-root", str(tmp_path), "--model", "fixture"]
    with pytest.raises(ValueError, match="disagrees"):
        dataset.annotate_main(args)
    assert len(common.read_json(output)) == 1
    fail[0] = False
    dataset.annotate_main(args + ["--resume"])
    assert len(common.read_json(output)) == 2
    assert calls == [rows[0]["local_image"], rows[1]["local_image"], rows[1]["local_image"]]
    dataset.annotate_main(args + ["--resume"])
    assert len(closed) == 2 and len(calls) == 3

    rows[0]["claim"] = "A different ceremony."
    common.write_json(source, rows)
    with pytest.raises(ValueError, match="differs from source labels"):
        dataset.annotate_main(args + ["--resume"])
    assert len(calls) == 3


def test_teacher_limit_zero_does_not_load_backend(tmp_path, monkeypatch):
    from src.backends import api_backend

    source, output = tmp_path / "source.json", tmp_path / "teacher.json"
    common.write_json(source, reviewed_pair(tmp_path))
    monkeypatch.setattr(api_backend, "APIBackend", lambda *args: pytest.fail("No paid work for --limit 0"))
    dataset.annotate_main(["--input", str(source), "--output", str(output), "--data-root", str(tmp_path),
                           "--model", "fixture", "--limit", "0"])
    assert common.read_json(output) == []


def test_split_rejects_shared_source_images_crossing_partitions(tmp_path):
    rows = []
    for i, year in enumerate([2010, 2011, 2019, 2020]):
        rows += reviewed_pair(tmp_path, f"seen{i}", "seen", year)
    for i in range(2):
        rows += reviewed_pair(tmp_path, f"ood{i}", "heldout", 2019)
    rows[4]["local_image"] = "nested/../" + rows[0]["local_image"]
    with pytest.raises(ValueError, match="Source image overlaps"):
        dataset.build_splits(rows, ood_target=1, val_frac=0)
    rows[4]["local_image"] = "seen2_original.png"
    rows[2]["local_image"] = "nested/../" + rows[0]["local_image"]
    result = dataset.build_splits(rows, ood_target=1, val_frac=0)
    assert len(result["train"]) == 4
