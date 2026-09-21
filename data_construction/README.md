# Recreate the dataset images

**The editing instructions and dataset text are already provided. Given an authentic image, the only model call needed is to apply its supplied editing prompt.** The `reproduce` command downloads the selected source images and performs that edit for you.

You do **not** need to regenerate prompts, choose edit types, rerun screening, generate teacher rationales, or rebuild the splits. The released metadata fixes 4,406 claims and their original/edited pairs. See [the data reference](../data/README.md) for the included files and annotation schema.

**Reproduction caveat:** image generation is stochastic. The same prompt and model may produce a slightly different image or a different realization of the edit; source URLs may also change or become unavailable. This recreates the dataset using the released instructions, but does not recover the exact image bytes or guarantee the paper's scores. Check new images against the supplied annotations before using them for training or evaluation. Historical human reviews apply to the original generations.

## 1. Install the dependencies

Run from the repository root, using Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This single requirements file includes dependencies for both GPT Image and FLUX.2 recreation, as well as inference and evaluation. GPT Image uses the hosted API; running FLUX.2 locally requires a CUDA GPU.

## 2. Recreate with GPT Image

Set your OpenAI API key, or put it in the repository's `.env` file using [.env.example](../.env.example):

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

Recreate all four main splits:

```bash
python -m data_construction reproduce --backend gpt-image
```

The command reads source URLs and prompts from `data/editing_prompts.json`, and membership and labels from the four main files in `data/splits/`. It downloads each original image, applies its supplied `edit_result.editing_prompt` with `gpt-image-1.5` at `medium` quality, and saves the original/edited pairs and matching manifests under `data/reproduced/gpt-image/`. It uses the original source image as the GPT Image input and normalizes the resulting dataset images to 768 × 512 pixels with letterboxing. API calls are billed to your account.

For a small first run, use:

```bash
python -m data_construction reproduce --backend gpt-image --limit 5
```

Then run the full command above to continue. Valid completed outputs are reused automatically; failed or missing images are retried. A limited run writes only the successfully completed pairs to its output split manifests.

Useful alternatives:

```bash
# Inspect the workload without downloads, model loading, or API calls.
python -m data_construction reproduce --dry-run

# Download the source images first, without generating edits.
python -m data_construction reproduce --download-only

# Recreate just the ID test set.
python -m data_construction reproduce --backend gpt-image --split test_id

# Reuse an existing directory of downloaded source images by filename.
python -m data_construction reproduce --source-image-dir /path/to/source_images
```

The source URLs are included, so the full upstream TARA metadata download and its `input/` folder are unnecessary for this workflow. If a source URL fails, provide the corresponding authentic image under the `source_image` filename recorded in `data/editing_prompts.json`, then rerun. The command reports failures instead of silently treating the missing pair as complete.

## Alternative: recreate with FLUX.2

To reproduce the alternative-generator test setting, use **FLUX.2-klein-9B** locally with a CUDA GPU. Its dependencies are included in `requirements.txt`, installed above. Obtain access to the model weights if required by the model provider, and run:

```bash
python -m data_construction reproduce --backend flux2 --split test_id
python -m data_construction reproduce --backend flux2 --split test_ood
```

Outputs go to `data/reproduced/flux2/`. This backend uses the same editing instructions, appends `Keep everything else the same in the image.`, and follows the experiment script's settings: `black-forest-labs/FLUX.2-klein-9B`, 28 steps, seed 42, and a 768 × 512 normalized source image. Add `--cpu-offload` to offload model components to CPU. Use `--limit 5` for a small run, or omit `--split` to recreate both released test sets (1,401 pairs) with FLUX.2. The released FLUX variant covers the test sets only.

The `flux2_editing_prompt` and `flux2_edited_image` fields are already included in the test records of `data/editing_prompts.json`. The command uses the main test labels and writes matching `splits/test_id.json` and `splits/test_ood.json` under its output root; no separate FLUX split files are needed in `data/splits/`.

GPT Image and FLUX.2 save to separate directories by default. When changing the model or generation settings, choose a new `--output-root` so the run has its own images and checkpoint. The implementation follows the [OpenAI image editing API](https://developers.openai.com/api/docs/guides/image-generation) and the [FLUX.2-klein-9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).

## 3. Use the recreated data

The recreated data root contains raw downloaded images under `images/source/`, normalized authentic images under `images/original/`, and edited images under `images/edited/gpt_image/` or `images/edited/flux2/`. Matching manifests are saved under `splits/`, with generation settings in `generation_config.json` and progress in `generation_results.json`. All image paths in the manifests are relative to that root. Only complete original/edited pairs with successfully generated images enter the output manifests; confirm the final counts before using a full split.

The supplied text annotations and teacher rationales describe the original experiment images. New output records are marked `review_required`; inspect each regenerated edit and update any mismatched visual evidence, explanation, or rationale. After that check, use the [training and evaluation environment](../README.md#environment) and run, for example:

```bash
python -m src.training.prepare \
  --format sft \
  --data-root data/reproduced/gpt-image
```

For FLUX.2 evaluation, pass `--data-root data/reproduced/flux2` and `--split test_id` or `--split test_ood`. The recreated manifests record the selected generator. You can also set `MIC_DATA_ROOT` once to use the standard commands in the [main README](../README.md#experiments).

Run `python -m data_construction reproduce --help` for all options. No original source images or generated image binaries are tracked in this repository.

## Optional: build a new dataset from scratch

The earlier construction stages remain available for extending the benchmark with new claims. Their intermediate source metadata, screening, verification, and generation files are not shipped and are **not required** to recreate the released images.

For a new dataset, obtain the full upstream TARA metadata from the [official repository](https://github.com/zeyofu/TARA) or the authors' [Google Drive folder](https://drive.google.com/drive/folders/1KNcEN3yvhki4XNIfg-t5mXlQZvS1h1XA?usp=sharing). The metadata files are `train.jsonl`, `gold_dev.jsonl`, and `gold_test.jsonl`. Images and source datasets remain subject to their providers' terms.

| Stage | Purpose |
| --- | --- |
| `prepare`, `download` | Import full upstream TARA metadata and download source images |
| `filter`, `verify` | Screen suitability and identify visible contextual cues |
| `propose` | Choose edit types and generate new editing instructions |
| `edit` | Apply newly proposed instructions |
| `review` | Export a human-review template or assemble approved pairs |
| `split` | Construct new entity/temporal train, validation, ID, and OOD splits |
| `annotate` | Generate optional teacher rationales |

Use `python -m data_construction STAGE --help` for a stage's arguments. For a new dataset, set explicit input/output paths in a separate directory: these stages' defaults target files under `data/`. The construction templates in [prompts.py](prompts.py) follow the paper's Prompts appendix. The supplied `data/editing_prompts.json` already contains the ready-to-run image instructions.
