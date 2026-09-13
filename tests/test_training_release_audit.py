"""Exercise the bundled GRPO media contract without loading the GPU stack."""

import ast
import copy
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image

from src.training import reward
from src.training.prepare import grpo_record, prepare

ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / "training/verl/verl"


def load_functions(path, names, class_name=None):
    """Run actual helper bodies, omitting imports that need torch/Ray/Transformers."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *functions], type_ignores=[])
    namespace = {"copy": copy, "re": re, "Image": Image, "BytesIO": BytesIO, "traceback": traceback}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


@pytest.fixture
def media_dataset(monkeypatch):
    def fetch_image(item, image_patch_size=14):
        # Exercise qwen-vl-utils' media input contract with Pillow instead of GPU dependencies.
        value = item["image"] if "image" in item else item["image_url"]
        return value if isinstance(value, Image.Image) else Image.open(value)

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(fetch_image=fetch_image))
    vision = load_functions(VERL / "utils/dataset/vision_utils.py", ["process_image"])
    monkeypatch.setitem(sys.modules, "verl.utils.dataset.vision_utils",
                        SimpleNamespace(**vision, process_video=None))
    methods = load_functions(VERL / "utils/dataset/rl_dataset.py",
                             ["_build_messages", "maybe_filter_out_long_prompts"], "RLHFDataset")
    dataset = type("IsolatedRLHFDataset", (), methods)()
    config = yaml.safe_load((ROOT / "configs/grpo/mic.yaml").read_text())["data"]
    dataset.__dict__.update(filter_overlong_prompts=True, tokenizer=None, prompt_key="prompt",
                            image_key="images", video_key="videos", max_prompt_length=100,
                            apply_chat_template_kwargs={}, tool_schemas=None, num_workers=1,
                            image_patch_size=config["image_patch_size"])
    return dataset


@pytest.mark.parametrize("store_image_bytes", [False, True], ids=["paths-only", "embedded"])
def test_grpo_parquet_images_reach_length_filter_and_rollout(tmp_path, media_dataset, store_image_bytes):
    records = []
    for size in (8, 128):
        path = tmp_path / f"image_{size}.png"
        Image.new("RGB", (size, size), "white").save(path)
        row = {"_id": str(size), "image_type": "original", "local_image": path.name, "claim": "A claim."}
        records.append(grpo_record(row, "train", tmp_path, store_image_bytes))
    parquet = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(records), parquet)
    records = pq.read_table(parquet).to_pylist()
    seen = []

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["type"] == "image"
            return "<image>A claim"

        def __call__(self, text, images, **kwargs):
            seen.append(images)
            return {"input_ids": [[0] * (5 + (images[0].width if images else 0))]}

    class Frame(list):
        def filter(self, predicate, **kwargs):
            return Frame(row for row in self if predicate(copy.deepcopy(row)))

    media_dataset.processor = Processor()
    kept = media_dataset.maybe_filter_out_long_prompts(Frame(records))
    assert len(kept) == 1, "Image tokens must count toward the prompt-length limit"
    assert [images[0].width for images in seen] == [8, 128]

    # The actual rollout consumer receives the same image after Parquet conversion.
    messages = media_dataset._build_messages(copy.deepcopy(kept[0]))
    content = messages[0]["content"][0]
    from qwen_vl_utils import fetch_image
    with fetch_image(content) as image:
        assert image.size == (8, 8)


def test_parallel_rewards_initialize_one_embedding_model(monkeypatch):
    created = []
    barrier = Barrier(4)

    class Embedder:
        def __init__(self, **kwargs):
            created.append(self)
            time.sleep(0.03)

        def cosine(self, prediction, target):
            return 1.0

    monkeypatch.setitem(sys.modules, "src.embedding", SimpleNamespace(Qwen3Embedder=Embedder))
    monkeypatch.setattr(reward, "_embedder", None)

    def score(_):
        barrier.wait(timeout=5)
        return reward._semantic_similarity("cue", "cue")

    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = list(executor.map(score, range(4)))
    assert scores == [1.0] * 4
    assert len(created) == 1


def test_training_rejects_reused_source_images_across_claim_ids(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "shared.png")
    for split in ("train", "val"):
        rows = []
        for image_type in ("original", "edited"):
            path = "shared.png" if image_type == "original" else f"{split}.png"
            if image_type == "edited":
                Image.new("RGB", (8, 8)).save(tmp_path / path)
            rows.append({"_id": split, "claim": f"Claim {split}", "image_type": image_type,
                         "local_image": path, "edit_type": "flag", "gt_entity_fine": "A flag",
                         "gt_why_contradicts": "The flag conflicts with the claim."})
        (tmp_path / f"{split}.json").write_text(json.dumps(rows))
    output = tmp_path / "converted"
    with pytest.raises(ValueError, match="Source image overlap"):
        prepare(tmp_path, tmp_path, output, "sft")
    assert not output.exists()
