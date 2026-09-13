"""Offline regressions for inference recovery and trustworthy score files."""

import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import run_infer, run_score
from src.backends import api_backend
from src.backends.vllm_backend import VLLMBackend, extract_verdict_inconsistent_prob
from src.metrics.verdict import score_one
from src.parsers.cot_tagged import parse
from src.records import read_predictions
from src.schema import EvalSample, Prediction


def prediction(edited=False):
    raw = ("<verdict>INCONSISTENT</verdict><type>flag</type>"
           "<visual>A flag</visual><explanation>The flag is from another country.</explanation>"
           if edited else "<verdict>CONSISTENT</verdict><type>None</type>"
           "<visual>None</visual><explanation>None</explanation>")
    parsed = parse(raw)
    return Prediction(sample_id=f"article__{'edited' if edited else 'original'}",
                      run_id="audit", model="fixture", split="test_id_edit",
                      prompt_variant="canonical", generator="fixture" if edited else "none",
                      raw_text=raw, parse_ok=parsed["parse_ok"], parse_errors=parsed["parse_errors"],
                      **{f"pred_{key}": parsed[key] for key in
                         ("verdict", "type", "visual", "explanation", "think")}).to_dict()


def write_predictions(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def samples(tmp_path, monkeypatch):
    result = []
    for edited in (False, True):
        pred = prediction(edited)
        image = tmp_path / f"{pred['sample_id']}.png"
        image.write_bytes(b"mock image read only by the fake backend")
        result.append(EvalSample(sample_id=pred["sample_id"], article_id="article",
                                 split="test_id_edit", image_path=str(image), claim="A claim",
                                 is_edited=edited, generator=pred["generator"],
                                 gt_edit_type="flag" if edited else None,
                                 gt_entity_fine=pred["pred_visual"],
                                 gt_why_contradicts=pred["pred_explanation"]))
    monkeypatch.setattr(run_score, "load_split", lambda *args: result)
    monkeypatch.setattr(run_infer, "load_split", lambda *args: result)
    return result


@pytest.mark.parametrize("invalid", [None, "", "UNKNOWN", "consistent"])
@pytest.mark.parametrize("gt", ["CONSISTENT", "INCONSISTENT"])
def test_invalid_verdict_is_always_wrong(gt, invalid):
    metrics = score_one(gt, invalid)
    assert metrics["verdict_correct"] == 0
    assert metrics["verdict_invalid"] == 1
    assert metrics["verdict_fp"] + metrics["verdict_fn"] == 1


@pytest.mark.parametrize("field,value", [("model", "other"), ("run_id", "other"),
                                         ("prompt_variant", "other"), ("split", "other")])
def test_prediction_file_cannot_mix_runs(tmp_path, field, value):
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [prediction(), dict(prediction(True), **{field: value})])
    with pytest.raises(ValueError, match="mixed"):
        read_predictions(path)


@pytest.mark.parametrize("change", [{"parse_ok": "false"}, {"raw_text": None},
                                    {"parse_errors": "failed"}, {"pred_visual": 12}])
def test_malformed_prediction_schema_fails(tmp_path, change):
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [dict(prediction(), **change)])
    with pytest.raises(ValueError):
        read_predictions(path)


def test_resume_rejects_corrupt_tail_and_metadata(tmp_path, samples):
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [prediction()])
    with pytest.raises(ValueError, match="different run/model"):
        run_infer._load_existing(path, expected={"model": "different"}, samples=samples)
    with pytest.raises(ValueError, match="outside"):
        run_infer._load_existing(path, samples=samples[1:])
    write_predictions(path, [dict(prediction(), inference_config={"model": "old"})])
    with pytest.raises(ValueError, match="different inference"):
        run_infer._load_existing(path, inference_config={"model": "new"})
    path.write_text(path.read_text() + '{"sample_id":', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid prediction JSON"):
        run_infer._load_existing(path)


def test_resume_requires_saved_inference_settings_when_expected(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [prediction()])
    assert run_infer._load_existing(path) == {"article__original"}
    with pytest.raises(ValueError, match="missing or different inference settings"):
        run_infer._load_existing(path, inference_config={"model": "fixture", "adapter_path": "new-adapter"})


def test_resume_without_trailing_newline_is_valid_jsonl(tmp_path, samples, monkeypatch):
    calls = []
    class FakeBackend:
        def __init__(self, **kwargs):
            pass
        def load(self):
            pass
        def unload(self):
            pass
        def run(self, image_path, prompt):
            calls.append(image_path)
            return prediction(True)["raw_text"]
    monkeypatch.setattr(api_backend, "APIBackend", FakeBackend)
    path = tmp_path / "audit.jsonl"
    config = {"model": "fixture", "backend": "api", "provider": "openai",
              "adapter_path": None, "max_model_len": 8192,
              "collect_verdict_logprobs": False, "generator_override": None}
    path.write_text(json.dumps(dict(prediction(), inference_config=config)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_infer", "--model", "fixture", "--backend", "api",
                                    "--provider", "openai", "--split", "test_id_edit",
                                    "--run-name", "audit", "--output-dir", str(tmp_path)])
    run_infer.main()
    assert len(calls) == 1
    assert len(read_predictions(path)) == 2
    run_infer.main()
    assert len(calls) == 1


@pytest.mark.parametrize("change", [{"raw_text": "garbage"}, {"pred_verdict": "UNKNOWN"},
                                    {"pred_visual": "Different extracted text"}])
def test_stale_or_tampered_parser_fields_receive_no_credit(tmp_path, samples, change):
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [prediction(), dict(prediction(True), **change)])
    embedder = SimpleNamespace(cosine_pairs=lambda pairs: [1.0] * len(pairs))
    scored, summary = run_score.score_run(path, "test_id_edit", embedder=embedder)
    assert scored[1].metrics["parse_failed_forced_zero"] == 1
    assert scored[1].metrics["verdict_correct"] == 0
    assert summary["type_acc"]["type_acc"] == 0
    assert summary["visual"]["visual"] == 0
    assert summary["explanation"]["explanation"] == 0


@pytest.mark.parametrize("filename,outname", [("preds.jsonl", "preds.jsonl"),
                                             ("preds.summary.json", "preds.jsonl")])
def test_scoring_cannot_overwrite_predictions_or_summary(tmp_path, monkeypatch, filename, outname):
    path = tmp_path / filename
    write_predictions(path, [prediction()])
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["run_score", "--predictions", str(path),
                                    "--split", "test_id_edit", "--out", str(tmp_path / outname)])
    with pytest.raises(SystemExit):
        run_score.main()
    assert path.read_bytes() == before


def test_nested_final_tags_do_not_parse():
    raw = prediction(True)["raw_text"].replace("<visual>A flag", "<visual>A <visual>flag")
    assert not parse(raw)["parse_ok"]


def test_verdict_confidence_uses_final_label_and_valid_prefixes():
    pieces = ["<think>", "<verdict>", "INCONSISTENT", "</verdict>", "</think>",
              "<verdict>", "\n", "CONSISTENT", "</verdict>", "Clearly", "Image"]
    tokenizer = SimpleNamespace(decode=lambda ids: "".join(pieces[i] for i in ids))
    output = SimpleNamespace(text="".join(pieces[:9]), token_ids=list(range(9)),
                             logprobs=[{2: -0.1, 7: -2.0}] * 7
                             + [{7: -1.0, 2: -2.0, 9: -0.01, 10: -0.02}, {}])
    score = extract_verdict_inconsistent_prob(output, tokenizer)
    assert score["verdict_token_step"] == 7
    assert score["p_inconsistent"] == pytest.approx(1 / (1 + math.e))


def test_vllm_uses_the_saved_adapter_rank(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text('{"r": 64, "rank_pattern": {}}')
    calls = []
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **kwargs: calls.append(kwargs)))
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", str(tmp_path / "miopen_db"))
    monkeypatch.setenv("MIOPEN_CUSTOM_CACHE_DIR", str(tmp_path / "miopen_cache"))
    VLLMBackend("fixture", adapter_path=str(tmp_path)).load()
    assert calls[0]["enable_lora"] is True
    assert calls[0]["max_lora_rank"] == 64


def test_nonfinite_similarity_does_not_become_perfect_credit(monkeypatch):
    # Isolate the numeric guard without installing or loading a model.
    torch = SimpleNamespace(Tensor=object, inference_mode=lambda: lambda function: function)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=None, AutoTokenizer=None))
    monkeypatch.delitem(sys.modules, "src.embedding", raising=False)
    module = importlib.import_module("src.embedding")
    try:
        for value in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(FloatingPointError, match="not finite"):
                module._clip_similarity(value)
        assert module._clip_similarity(0.25) == 0.25
        assert module._clip_similarity(-0.01) == 0.0
    finally:
        sys.modules.pop("src.embedding", None)
