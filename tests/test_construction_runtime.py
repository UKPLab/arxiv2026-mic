"""Offline regression checks for construction checkpoints and stage contracts."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
from PIL import Image

from data_construction import common, dataset, images


@pytest.fixture
def client_factory(monkeypatch):
    clients = []

    def create():
        client = SimpleNamespace(closed=False)
        client.close = lambda: setattr(client, "closed", True)
        clients.append(client)
        return client

    monkeypatch.setattr(common, "create_client", create)
    return clients


def test_resume_retries_failures_and_replaces_duplicate_checkpoint_ids(tmp_path, client_factory):
    output = tmp_path / "results.json"
    common.write_json(output, [
        {"_id": "a", "result": {"error": "old failure"}},
        {"_id": "b", "result": {"value": "keep"}},
        {"_id": "a", "result": {"error": "latest failure"}},
    ])
    calls = []

    def process(row, client):
        calls.append(row["_id"])
        return dict(row, result={"value": "retried"})

    rows = common.run_api_batch(
        [{"_id": "b"}, {"_id": "a"}, {"_id": "c"}], process, output,
        result_key="result", workers=2, limit=1, resume=True,
    )
    assert calls == ["a"]
    assert rows == common.read_json(output) == [
        {"_id": "a", "result": {"value": "retried"}},
        {"_id": "b", "result": {"value": "keep"}},
    ]
    assert client_factory[0].closed


@pytest.mark.parametrize("resume", [False, True])
def test_empty_batches_do_not_create_a_client(tmp_path, monkeypatch, resume):
    output = tmp_path / "results.json"
    saved = [{"_id": "a", "result": {"ok": True}}]
    common.write_json(output, saved)

    def unexpected_call(*args):
        pytest.fail("A fully resumed batch or --limit 0 must not make API requests")

    monkeypatch.setattr(common, "create_client", unexpected_call)
    rows = common.run_api_batch(
        [{"_id": "a"}], unexpected_call, output, result_key="result",
        limit=None if resume else 0, resume=resume,
    )
    assert rows == common.read_json(output) == (saved if resume else [])


def test_worker_errors_are_checkpointed_without_losing_successes(tmp_path, client_factory):
    def process(row, client):
        if row["_id"] == "b":
            raise RuntimeError("temporary service failure")
        return dict(row, result={"ok": True})

    output = tmp_path / "nested/results.json"
    common.run_api_batch(
        [{"_id": "b", "caption": "Preserve this claim"}, {"_id": "a"}],
        process, output, result_key="result", workers=2, save_every=1,
    )
    assert common.read_json(output) == [
        {"_id": "a", "result": {"ok": True}},
        {"_id": "b", "caption": "Preserve this claim", "result": {"error": "temporary service failure"}},
    ]
    assert client_factory[0].closed


def test_interruption_saves_completed_work_and_closes_client(tmp_path, monkeypatch, client_factory):
    def interrupt_after_first(futures):
        futures = list(futures)
        for future in futures:
            future.result()
        yield futures[0]
        raise KeyboardInterrupt

    monkeypatch.setattr(common, "as_completed", interrupt_after_first)
    output = tmp_path / "results.json"
    with pytest.raises(KeyboardInterrupt):
        common.run_api_batch(
            [{"_id": "a"}, {"_id": "b"}], lambda row, client: dict(row, result={"ok": True}),
            output, result_key="result", workers=2,
        )
    assert [row["_id"] for row in common.read_json(output)] == ["a", "b"]
    assert client_factory[0].closed


def test_worker_interrupt_does_not_bypass_saving_other_completed_records(tmp_path, monkeypatch, client_factory):
    def process(row, client):
        if row["_id"] == "b":
            raise KeyboardInterrupt
        return dict(row, result={"ok": True})

    def completed_in_submission_order(futures):
        for future in futures:
            future.exception()  # Wait without re-raising the worker's exception.
        yield from futures

    monkeypatch.setattr(common, "as_completed", completed_in_submission_order)
    output = tmp_path / "results.json"
    with pytest.raises(KeyboardInterrupt):
        common.run_api_batch(
            [{"_id": "a"}, {"_id": "b"}], process, output, result_key="result", workers=2,
        )
    assert output.exists(), "A worker interruption must not prevent saving completed paid work"
    assert any(row["_id"] == "a" and row["result"] == {"ok": True} for row in common.read_json(output))
    assert client_factory[0].closed


def test_interruption_retains_in_flight_work_finishing_during_shutdown(tmp_path, monkeypatch, client_factory):
    in_flight, shutting_down = Event(), Event()

    class ControlledExecutor(ThreadPoolExecutor):
        def __exit__(self, *args):
            shutting_down.set()
            return super().__exit__(*args)

    def process(row, client):
        if row["_id"] == "b":
            in_flight.set()
            assert shutting_down.wait(5), "The running request should finish during executor shutdown"
        return dict(row, result={"ok": True})

    def interrupt_with_one_request_running(futures):
        first = next(iter(futures))
        first.result()
        assert in_flight.wait(5)
        yield first
        raise KeyboardInterrupt

    monkeypatch.setattr(common, "ThreadPoolExecutor", ControlledExecutor)
    monkeypatch.setattr(common, "as_completed", interrupt_with_one_request_running)
    output = tmp_path / "results.json"
    with pytest.raises(KeyboardInterrupt):
        common.run_api_batch(
            [{"_id": "a"}, {"_id": "b"}], process, output, result_key="result", workers=2,
        )
    assert common.read_json(output) == [
        {"_id": "a", "result": {"ok": True}}, {"_id": "b", "result": {"ok": True}},
    ]
    assert client_factory[0].closed


@pytest.mark.parametrize("records,kwargs", [
    ([{"_id": "a"}, {"_id": "a"}], {}),
    ([{"_id": "a"}], {"workers": 0}),
    ([{"_id": "a"}], {"limit": -1}),
])
def test_invalid_batch_inputs_fail_before_output_or_client(tmp_path, monkeypatch, records, kwargs):
    monkeypatch.setattr(common, "create_client", lambda: pytest.fail("Invalid input must fail before API setup"))
    output = tmp_path / "results.json"
    with pytest.raises(ValueError):
        common.run_api_batch(records, lambda row, client: row, output, result_key="result", **kwargs)
    assert not output.exists()


def test_atomic_json_write_preserves_checkpoint_on_serialization_failure(tmp_path):
    output = tmp_path / "results.json"
    common.write_json(output, [{"claim": "原始记录"}])
    before = output.read_bytes()
    with pytest.raises(TypeError):
        common.write_json(output, [{"claim": "partial write"}, {"unsupported": object()}])
    assert output.read_bytes() == before
    assert list(tmp_path.iterdir()) == [output]


def test_metadata_preserves_dates_and_normalizes_source_fields():
    record = {
        "_id": "a", "year": 2019, "date": "2019-04-05", "time": "April 2019",
        "image_url": "https://example.com/images/2001/02/03/example.jpg",
        "headline": {"main": "Source headline"}, "keywords": [{"value": "flag"}, "ceremony"],
    }
    result = common.metadata(record)
    assert (result["year"], result["date"], result["time"]) == (2019, "2019-04-05", "April 2019")
    assert result["headline"] == "Source headline"
    assert result["keywords"] == ["flag", "ceremony"]
    derived = common.metadata({"_id": "b", "image_url": record["image_url"]})
    assert (derived["year"], derived["date"]) == ("2001", "2001-02-03")
    without_url_date = common.metadata(dict(record, image_url="https://example.com/photo.jpg"))
    assert (without_url_date["year"], without_url_date["date"]) == (2019, "2019-04-05")


@pytest.mark.parametrize("model,with_image,settings", [
    ("gpt-4o-mini", True, {"max_tokens": 400, "temperature": 0}),
    ("gpt-4o", False, {"max_tokens": 500, "temperature": 0.7}),
    ("gpt-5", True, {"max_completion_tokens": 16000}),
])
def test_chat_json_preserves_model_parameters_and_image_encoding(tmp_path, model, with_image, settings):
    image = tmp_path / "source.PNG"
    Image.new("RGB", (8, 8), "white").save(image)
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"suitable": true}'))],
            usage=SimpleNamespace(prompt_tokens=17, completion_tokens=4),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = common.chat_json(
        client, model, "Inspect this image.", image_path=image if with_image else None,
        detail="low", max_tokens=400 if model == "gpt-4o-mini" else 500,
        temperature=0.7 if not with_image else 0,
    )
    assert result == {"suitable": True, "tokens": {"input": 17, "output": 4}}
    assert {key: seen[key] for key in ("max_tokens", "max_completion_tokens", "temperature") if key in seen} == settings
    assert seen["response_format"] == {"type": "json_object"}
    content = seen["messages"][0]["content"]
    if with_image:
        uri = content[0]["image_url"]
        assert uri["detail"] == "low"
        assert uri["url"].startswith("data:image/png;base64,")
        assert base64.b64decode(uri["url"].split(",", 1)[1]) == image.read_bytes()
        assert content[1] == {"type": "text", "text": "Inspect this image."}
    else:
        assert content == "Inspect this image."


def test_prompt_assignments_and_region_hints_match_after_resuming(tmp_path, monkeypatch, client_factory):
    source = tmp_path / "round2.json"
    rows = [{
        "_id": name, "caption": f"Claim {name}", "year": 2019,
        "round2_result": {"quality": "high", "visual_description": "Visible flags and clothing",
                          "feasible_types": {kind: {"visible": True, "evidence": f"Visible {kind}"} for kind in types}},
    } for name, types in [
        ("e", ["flag", "clothing"]), ("b", ["flag"]), ("a", ["flag"]),
        ("d", ["clothing"]), ("c", ["flag", "clothing"]),
    ]]
    common.write_json(source, rows)
    calls = []

    def chat(client, model, prompt, **kwargs):
        calls.append(prompt)
        return {"editing_prompt": prompt}

    monkeypatch.setattr(images, "chat_json", chat)
    full, resumed = tmp_path / "full.json", tmp_path / "resumed.json"
    args = ["--model", "fixture", "--round2-path", str(source), "--max-workers", "2"]
    images.propose_main(args + ["--output", str(full)])
    images.propose_main(args + ["--output", str(resumed), "--limit", "2"])
    assert len(common.read_json(resumed)) == 2
    # Input order must not influence type assignments or region rotation.
    common.write_json(source, list(reversed(rows)))
    images.propose_main(args + ["--output", str(resumed), "--resume"])
    assert common.read_json(resumed) == common.read_json(full)
    assert len(calls) == 2 * len(rows)
    flags = [row for row in common.read_json(full) if row["edit_type"] == "flag"]
    assert len({row["edit_result"]["region_hint"] for row in flags}) == len(flags)
    assert all(client.closed for client in client_factory)


def test_edit_stage_keeps_shared_images_distinct_and_retries_missing_outputs(tmp_path, monkeypatch):
    source = tmp_path / "shared.png"
    Image.new("RGB", (8, 8), "white").save(source)
    prompts, results, output_dir = tmp_path / "prompts.json", tmp_path / "results.json", tmp_path / "edited"
    rows = [{
        "_id": name, "caption": f"Claim {name}", "local_image": source.name, "year": 2019,
        "edit_type": "flag", "edit_result": {"editing_prompt": f"Replace the flag for {name}.",
        "what_changed": "Flag changed", "why_contradicts": "Wrong flag", "region_hint": "Europe"},
    } for name in ("claim-a", "claim-b")]
    common.write_json(prompts, rows)
    calls, closed = [], []

    def edit(**kwargs):
        calls.append(kwargs["prompt"])
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(source.read_bytes()).decode())])

    monkeypatch.setattr(common, "create_client", lambda: SimpleNamespace(
        images=SimpleNamespace(edit=edit), close=lambda: closed.append(True),
    ))
    args = ["--prompts", str(prompts), "--results", str(results), "--image-dir", str(tmp_path),
            "--output-dir", str(output_dir), "--max-workers", "2"]
    images.edit_main(args)
    saved = common.read_json(results)
    paths = [output_dir / row["edit_status"]["output_image"] for row in saved]
    assert len(set(paths)) == 2 and all(path.is_file() for path in paths)
    assert all(row["year"] == 2019 and row["editing_prompt"] == row["edit_result"]["editing_prompt"] for row in saved)
    assert all(row["generator"] == "gpt-image-1.5" and row["region_hint"] == "Europe" for row in saved)
    images.edit_main(args + ["--resume"])
    assert len(calls) == 2
    paths[0].unlink()
    images.edit_main(args + ["--resume"])
    assert len(calls) == 3 and calls[-1] == rows[0]["edit_result"]["editing_prompt"]
    assert common.read_json(results) == saved
    assert len(closed) == 2


def test_mocked_construction_stages_preserve_reviewable_pair_schema(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(source)
    metadata_path, screen, verified, prompts, edits = [
        tmp_path / f"{name}.json" for name in ("metadata", "screen", "verified", "prompts", "edits")
    ]
    common.write_json(metadata_path, [{
        "_id": "news/claim", "caption": "A ceremony in 2019.", "year": 2019, "date": "2019-05-01",
        "local_image": source.name, "headline": {"main": "Ceremony"}, "keywords": [{"value": "flag"}],
    }])
    responses = iter([
        {"suitable": True, "feasible_types": ["flag"]},
        {"quality": "high", "visual_description": "A clearly visible flag",
         "feasible_types": {"flag": {"visible": True, "evidence": "A flag beside the podium"}}},
        {"editing_prompt": "Replace only the flag.", "what_changed": "Original flag to replacement flag",
         "why_contradicts": "The replacement flag conflicts with the event."},
    ])
    closed = []

    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(responses))))], usage=None,
        )

    monkeypatch.setattr(common, "create_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        images=SimpleNamespace(edit=lambda **kwargs: SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(source.read_bytes()).decode())],
        )), close=lambda: closed.append(True),
    ))
    images.filter_main(["--metadata", str(metadata_path), "--image-dir", str(tmp_path),
                        "--output", str(screen), "--min-width", "0", "--min-height", "0"])
    images.verify_main(["--round1-path", str(screen), "--image-dir", str(tmp_path),
                        "--output", str(verified), "--model", "fixture"])
    images.propose_main(["--round2-path", str(verified), "--output", str(prompts), "--model", "fixture"])
    images.edit_main(["--prompts", str(prompts), "--image-dir", str(tmp_path),
                      "--output-dir", str(tmp_path / "edited"), "--results", str(edits)])
    generated = common.read_json(edits)
    review = dataset.review_template(generated)
    assert review[0]["year"] == 2019
    assert review[0]["proposed_change"] == "Original flag to replacement flag"
    review[0].update(accepted=True, gt_entity_fine="The replacement flag", gt_entity_canonical="replacement_flag",
                     gt_why_contradicts="The flag conflicts with the ceremony.")
    original, edited = dataset.assemble(generated, review, tmp_path, ".", "edited")
    assert original["claim"] == edited["claim"] == "A ceremony in 2019."
    assert original["image_type"] == "original" and original["gt_entity_fine"] is None
    assert edited["image_type"] == "edited" and edited["edit_type"] == "flag"
    assert edited["generator"] == "gpt-image-1.5" and edited["year"] == 2019
    assert len(closed) == 4
