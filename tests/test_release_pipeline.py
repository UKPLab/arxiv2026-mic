"""Offline checks for the public data, training, and evaluation contracts."""

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from PIL import Image

from data_construction.dataset import assemble, review_template
from data_construction.images import prepare as prepare_tara
from src.data_loader import load_split
from src.parsers.cot_tagged import parse
from src.records import normalize_record, read_records, sample_id
from src.run_score import score_run
from src.training.prepare import build_target, prepare
from src.training import reward

ROOT = Path(__file__).resolve().parents[1]


def pair(root, article_id, entity="seen", year=2010, typ="flag"):
    rows = []
    for image_type in ("original", "edited"):
        path = root / f"{article_id}_{image_type}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        rows.append(normalize_record({
            "_id": article_id, "claim": f"A claim about {article_id}.",
            "image_type": image_type, "local_image": path.name,
            "generator": "gpt-image-1.5" if image_type == "edited" else "none",
            "year": year, "edit_type": typ, "gt_entity_fine": "A visible flag",
            "gt_entity_canonical": entity, "gt_why_contradicts": "This flag conflicts with the claim.",
        }))
    return rows


@pytest.fixture
def dataset(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir(parents=True)
    for name, article_id in (("train", "a"), ("val", "b"), ("test_id_edit", "c")):
        (splits / f"{name}.json").write_text(json.dumps(pair(tmp_path, article_id)))
    return tmp_path, splits


def predictions(root, split, output):
    rows = read_records(root / f"splits/{split}.json")
    preds = []
    for row in rows:
        raw = build_target(row)
        parsed = parse(raw)
        preds.append({
            "sample_id": sample_id(row), "run_id": "inference_test", "model": "fixture",
            "split": split, "prompt_variant": "canonical", "generator": row["generator"],
            "raw_text": raw, "parse_ok": parsed["parse_ok"], "parse_errors": parsed["parse_errors"],
            **{f"pred_{key}": parsed[key] for key in ("verdict", "type", "visual", "explanation")},
        })
    output.write_text("".join(json.dumps(p) + "\n" for p in preds))
    return preds


def test_sft_and_grpo_share_release_schema(dataset, tmp_path):
    root, splits = dataset
    prepare(splits, root, tmp_path / "sft", "sft")
    prepare(splits, root, tmp_path / "grpo", "grpo")
    sft = json.loads((tmp_path / "sft/train.json").read_text())
    rl = pq.read_table(tmp_path / "grpo/train.parquet").to_pylist()
    assert len(sft) == len(rl) == 2
    assert sft[1]["messages"][0]["content"] == rl[1]["prompt"][0]["content"]
    gt = json.loads(rl[1]["reward_model"]["ground_truth"])
    assert gt["gt_entity_fine"] == parse(sft[1]["messages"][1]["content"])["visual"]
    assert rl[1]["data_source"] == "mic"
    with Image.open(io.BytesIO(rl[1]["images"][0]["bytes"])) as image:
        assert image.size == (8, 8)
    assert json.loads((tmp_path / "sft/dataset_info.json").read_text())["mic_train"]["file_name"] == "train.json"
    assert not json.loads((tmp_path / "sft/manifest.json").read_text())["test_data_read"]


def test_training_rejects_claim_leakage_before_writing(dataset, tmp_path):
    root, splits = dataset
    (splits / "val.json").write_text((splits / "train.json").read_text())
    with pytest.raises(ValueError, match="overlap"):
        prepare(splits, root, tmp_path / "bad", "sft")
    assert not (tmp_path / "bad").exists()


def test_missing_images_and_unpaired_records_fail(dataset, tmp_path):
    root, splits = dataset
    rows = json.loads((splits / "train.json").read_text())
    (root / rows[0]["local_image"]).unlink()
    with pytest.raises(FileNotFoundError):
        prepare(splits, root, tmp_path / "bad", "grpo")


def test_legacy_labels_map_on_input(tmp_path):
    rows = pair(tmp_path, "legacy", typ="social_behavior")
    assert rows[1]["edit_type"] == "gesture"
    assert parse(build_target(rows[1]))["type"] == "gesture"


def test_parser_rejects_missing_duplicate_and_contradictory_fields(tmp_path):
    original, edited = pair(tmp_path, "parser")
    valid = build_target(edited)
    assert parse(valid)["parse_ok"]
    assert parse(valid.split("<think>", 1)[1])["parse_ok"]
    assert not parse(valid + "<verdict>CONSISTENT</verdict>")["parse_ok"]
    assert not parse("<verdict>CONSISTENT</verdict>")["parse_ok"]
    assert not parse(build_target(original).replace("<visual>None", "<visual>A flag"))["parse_ok"]
    assert not parse(valid.replace("<verdict>INCONSISTENT", "<verdict>CONSISTENT or INCONSISTENT"))["parse_ok"]


def test_component_reward_uses_correct_reference_fields(tmp_path, monkeypatch):
    original, edited = pair(tmp_path, "reward")
    calls = []
    def similarity(prediction, reference):
        calls.append((prediction, reference))
        return 1.0 if prediction == reference else 0.0
    monkeypatch.setattr(reward, "_semantic_similarity", similarity)
    gt = dict(verdict="INCONSISTENT", edit_type="flag", gt_entity_fine=edited["gt_entity_fine"],
              gt_why_contradicts=edited["gt_why_contradicts"])
    assert reward.compute_score("mic", build_target(edited), json.dumps(gt)) == pytest.approx(1.0)
    assert calls == [(edited["gt_entity_fine"], edited["gt_entity_fine"]),
                     (edited["gt_why_contradicts"], edited["gt_why_contradicts"])]
    assert reward.score_response(build_target(original), '{"verdict":"CONSISTENT"}') == pytest.approx(0.45)
    # Preserve the soft-format, component-independent policy.
    assert reward.score_response(build_target(edited).replace("<think>", ""), json.dumps(gt)) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        reward.score_response(build_target(edited), '{}')


def test_full_scoring_and_coverage_guards(dataset, tmp_path):
    root, _ = dataset
    pred_path = tmp_path / "predictions.jsonl"
    preds = predictions(root, "test_id_edit", pred_path)
    embedder = SimpleNamespace(cosine_pairs=lambda pairs: [1.0] * len(pairs))
    scored, agg = score_run(pred_path, "test_id_edit", embedder, data_root=root)
    assert agg["table1"]["macro_f1"] == 1.0
    assert agg["visual"]["visual"] == 1.0
    assert agg["coverage"] == {"expected": 2, "scored": 2, "missing": 0}
    _, skipped = score_run(pred_path, "test_id_edit", data_root=root)
    assert skipped["visual"]["visual"] is None
    assert skipped["explanation"]["explanation"] is None
    pred_path.write_text(json.dumps(preds[0]) + "\n")
    with pytest.raises(ValueError, match="Missing 1"):
        score_run(pred_path, "test_id_edit", data_root=root)
    _, partial = score_run(pred_path, "test_id_edit", data_root=root, allow_partial=True)
    assert partial["coverage"]["missing"] == 1
    pred_path.write_text(json.dumps(dict(preds[0], sample_id="unknown")) + "\n")
    with pytest.raises(ValueError, match="Unknown"):
        score_run(pred_path, "test_id_edit", embedder, data_root=root, allow_partial=True)
    pred_path.write_text((json.dumps(preds[0]) + "\n") * 2)
    with pytest.raises(ValueError, match="unique"):
        score_run(pred_path, "test_id_edit", data_root=root, allow_partial=True)


def test_malformed_prediction_gets_no_task_credit(dataset, tmp_path):
    root, _ = dataset
    path = tmp_path / "bad.jsonl"
    preds = predictions(root, "test_id_edit", path)
    preds[1]["parse_ok"] = False
    path.write_text("".join(json.dumps(p) + "\n" for p in preds))
    scored, summary = score_run(path, "test_id_edit", SimpleNamespace(cosine_pairs=lambda p: [1.0]*len(p)), data_root=root)
    assert scored[1].metrics["verdict_correct"] == 0
    assert scored[1].metrics["visual_score"] == 0
    assert summary["type_acc"]["type_acc"] == 0


def test_tara_import_rejects_conflicting_claims(tmp_path):
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    record = {"_id": "a", "image_url": "https://example.com/image.jpg", "caption": "A claim"}
    first.write_text(json.dumps(record) + "\n")
    second.write_text(json.dumps(dict(record, caption="Different claim")) + "\n")
    assert len(prepare_tara([first, first])) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_tara([first, second])


def test_review_gate_preserves_pairs(tmp_path):
    pair(tmp_path, "review")
    results = [{"_id": "review", "caption": "A claim", "local_image": "review_original.png",
                "edit_type": "flag", "generator": "gpt-image-1.5",
                "edit_status": {"output_image": "review_edited.png"}}]
    review = review_template(results)
    with pytest.raises(ValueError, match="No approved"):
        assemble(results, review, tmp_path, ".", ".")
    review[0].update(accepted=True, year=2019, gt_entity_canonical="flag_a",
                     gt_entity_fine="A flag", gt_why_contradicts="Wrong flag")
    rows = assemble(results, review, tmp_path, ".", ".")
    assert len(rows) == 2
    assert rows[0]["claim"] == rows[1]["claim"]
    assert rows[0]["gt_entity_fine"] is None


def test_edit_api_with_mocked_image_response(tmp_path):
    from data_construction.images import edit_one
    import base64
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    seen = {}
    def edit(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(image.read_bytes()).decode())])
    client = SimpleNamespace(images=SimpleNamespace(edit=edit))
    item = {"_id": "edit", "local_image": image.name, "edit_result": {"editing_prompt": "Replace the flag."}}
    _, status = edit_one(client, item, str(tmp_path), str(tmp_path / "edited"))
    assert "error" not in status
    assert seen["model"] == "gpt-image-1.5" and seen["size"] == "auto"
    assert (tmp_path / "edited" / status["output_image"]).is_file()


def test_entity_temporal_split_keeps_claims_together(tmp_path):
    rows = []
    for i, year in enumerate([2008, 2010, 2011, 2012, 2019, 2020]):
        rows += pair(tmp_path, f"seen{i}", "seen", year)
    for i in range(2):
        rows += pair(tmp_path, f"ood{i}", "heldout", 2019)
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(rows))
    output = tmp_path / "split"
    subprocess.run([sys.executable, "-m", "data_construction", "split", "--annotations", str(annotations),
                    "--output-dir", str(output), "--ood_target", "1"], cwd=ROOT, check=True, capture_output=True)
    manifests = {name: json.loads((output/f"{name}.json").read_text()) for name in ("train", "val", "test_id_edit", "test_ood_edit")}
    owners = {}
    for name, manifest in manifests.items():
        for row in manifest:
            assert owners.setdefault(row["_id"], name) == name
            assert (row["year"] < 2018) == (name in {"train", "val"})
    train_entities = {r["gt_entity_canonical"] for r in manifests["train"]}
    assert {r["gt_entity_canonical"] for r in manifests["test_id_edit"]} <= train_entities
    assert not {r["gt_entity_canonical"] for r in manifests["test_ood_edit"]} & train_entities


@pytest.mark.parametrize("command", [
    ("src.run_infer",), ("src.run_score",), ("src.training.prepare",),
    *(("data_construction", stage) for stage in (
        "prepare", "download", "filter", "verify", "propose", "edit", "review", "split", "annotate",
    )),
], ids=lambda command: " ".join(command))
def test_entrypoint_help(command):
    result = subprocess.run([sys.executable, "-m", *command, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_inference_resume_and_fresh_run(dataset, tmp_path, monkeypatch):
    from src import run_infer
    from src.backends import api_backend
    root, _ = dataset
    rows = read_records(root / "splits/test_id_edit.json")
    targets = {row["local_image"]: build_target(row) for row in rows}
    calls = []
    class FakeBackend:
        def __init__(self, **kwargs):
            pass
        def load(self):
            pass
        def unload(self):
            pass
        def run(self, path, prompt):
            calls.append(path)
            return targets[Path(path).name]
    monkeypatch.setattr(api_backend, "APIBackend", FakeBackend)
    argv = ["run_infer", "--model", "fixture", "--backend", "api", "--provider", "openai", "--api-workers", "1",
            "--split", "test_id_edit", "--run-name", "inference_test", "--data-root", str(root), "--output-dir", str(tmp_path / "preds")]
    monkeypatch.setattr(sys, "argv", argv)
    run_infer.main()
    run_infer.main()
    assert len(calls) == 2
    monkeypatch.setattr(sys, "argv", argv + ["--no-skip-existing"])
    run_infer.main()
    saved = [json.loads(line) for line in (tmp_path / "preds/inference_test.jsonl").read_text().splitlines()]
    assert len(calls) == 4 and len(saved) == 2
    assert [r["generator"] for r in saved] == ["none", "gpt-image-1.5"]


def test_grpo_config_composes_with_bundled_framework(tmp_path):
    import shutil
    from hydra import compose, initialize_config_dir
    dest = tmp_path / "config"
    shutil.copytree(ROOT / "training/verl/verl/trainer/config", dest)
    text = (ROOT / "configs/grpo/mic.yaml").read_text()
    # Relocate only the search path for CPU validation; preserve the actual configs.
    text = text.replace("hydra:\n  searchpath:\n    - pkg://verl.trainer.config\n\n", "")
    (dest / "mic.yaml").write_text(text)
    with initialize_config_dir(version_base=None, config_dir=str(dest)):
        config = compose(config_name="mic", overrides=["actor_rollout_ref.model.path=/tmp/fixture",
            "data.train_files=/tmp/train.parquet", "data.val_files=/tmp/val.parquet"])
    assert config.algorithm.adv_estimator == "grpo"
    assert config.actor_rollout_ref.actor.strategy == "fsdp"
    assert config.reward.custom_reward_function.name == "compute_score"
    assert config.actor_rollout_ref.rollout.n == 8
