# MIC-Bench data

This directory contains the editing instructions and fixed dataset annotations. Use the supplied prompts to recreate the images.

## Included files

```text
data/
├── editing_prompts.json         # Source URLs, editing prompts, types, and image paths
└── splits/
    ├── train.json
    ├── val.json
    ├── test_id.json
    └── test_ood.json
```

`editing_prompts.json` contains one record per claim. GPT Image instructions are in `edit_result.editing_prompt`; test records also include `flux2_editing_prompt`. The split files contain the paired original/edited records and their annotations.

## Dataset size

| Main split | Claims / image pairs | Image–claim records |
| --- | ---: | ---: |
| Train | 2,561 | 5,122 |
| Validation | 444 | 888 |
| Test ID | 703 | 1,406 |
| Test OOD | 698 | 1,396 |
| **Total** | **4,406** | **8,812** |

## Recreate the images

After completing the [dependency and API setup](../data_construction/README.md), run from the repository root:

```bash
# Recreate all four splits with GPT Image (uses paid API calls).
python -m data_construction reproduce --backend gpt-image

# Alternative: recreate the test sets with FLUX.2.
python -m data_construction reproduce --backend flux2
```

Add `--dry-run` to inspect the workload without downloading images, loading models, or making API calls. GPT Image covers all 4,406 pairs; FLUX.2 covers the 1,401 test pairs. Outputs are saved under `data/reproduced/gpt-image/` or `data/reproduced/flux2/`.

**Regenerated images may differ from the paper's images. Check them against the supplied annotations before training or evaluation.** See the [recreation guide](../data_construction/README.md) for image reuse, resuming, and output details.

## Annotation schema

Each split is a JSON array. An original and its edited counterpart share the same `_id` and claim. Image paths are relative to the data root.

| JSON field | Meaning |
| --- | --- |
| `_id`, `claim` | Claim identifier and text |
| `image_type` | `original` or `edited`; determines the reference verdict |
| `local_image` | Image path relative to the data root |
| `generator` | Image generation model, or `none` for originals |
| `edit_type` | Edit category; maps to `<type>` for edited images |
| `gt_entity_fine` | Description of the actual visible replacement; supplies `<visual>` |
| `gt_why_contradicts` | Reference explanation of the contradiction; supplies the training target's `<explanation>` |
| `gt_entity_canonical`, `year` | Entity identifier and year used for entity/temporal splitting |
| `cot_response` | Tagged reasoning and answer text |

For originals, the visual and explanation references and `gt_entity_canonical` are null. Their `edit_type` retains the paired edit's category, while the output uses `CONSISTENT` and `None` for `<type>`, `<visual>`, and `<explanation>`.

Every released `cot_response` contains these five tags (illustrative example):

```xml
<think>The visible flag does not match the flag identified in the claim.</think>
<verdict>INCONSISTENT</verdict>
<type>flag</type>
<visual>A red flag with five yellow stars.</visual>
<explanation>The claim identifies the flag as Japanese, but the visible design is the Chinese national flag.</explanation>
```

The `<visual>` text is filled from `gt_entity_fine` (`None` for originals). The explanation in `cot_response` can differ from `gt_why_contradicts`; training conversion uses the structured reference fields for the final answer.

## Type labels

`clothing`, `flag`, `gesture`, `signage`, `architecture`, `infrastructure`, `technology`, `branding`, `environment`.

See the main README for [training](../README.md#training) and [evaluation](../README.md#inference-and-evaluation).
