# Data

See the [construction overview](../data_construction/README.md) for the dataset workflow. This page describes the resulting files and labels.

## Source download

TARA (ACL 2022) is available from its [official repository](https://github.com/zeyofu/TARA) and the authors' [Google Drive folder](https://drive.google.com/drive/folders/1KNcEN3yvhki4XNIfg-t5mXlQZvS1h1XA?usp=sharing). Download the JSONL metadata under `input/`. The upstream code refers to `train.jsonl`, `gold_dev.jsonl`, and `gold_test.jsonl`. Use the actual files in your download; the repository does not redistribute the news images.

TARA is the starting corpus, not the finished MIC-Bench release. The manuscript describes 4,406 selected claims and 8,812 image–claim instances. No public MIC-Bench archive or checkpoint URL has been configured in this checkout. Re-running a hosted image generator does not guarantee the exact images or final counts used in the manuscript.

## Local layout

```text
data/
├── TARA/
│   ├── input/                 # Downloaded source metadata
│   ├── images/                # Authentic images
│   ├── edited_images/         # Generated counterparts
│   ├── edit_prompts.json
│   ├── edit_results.json
│   └── reviews.json
├── annotations.json          # Reviewed original/edited records
└── splits/
    ├── train.json
    ├── val.json
    ├── test_id_edit.json
    └── test_ood_edit.json
```

Teacher rationales can be stored in separate training and validation manifests before SFT conversion.

Each split is a JSON array. An edited record has the following structure (illustrative schema, not a benchmark instance):

```json
{
  "_id": "example-claim-id",
  "claim": "The accompanying factual claim.",
  "image_type": "edited",
  "local_image": "TARA/edited_images/example_edited.png",
  "generator": "gpt-image-1.5",
  "year": 2019,
  "edit_type": "flag",
  "gt_entity_fine": "The specific flag visible after editing.",
  "gt_entity_canonical": "A normalized entity identifier shared across equivalent flags.",
  "gt_why_contradicts": "The factual reason this flag contradicts the claim."
}
```

The paired original uses the same `_id` and claim, `image_type: "original"`, its authentic image path, and `generator: "none"`. Its inconsistency-specific reference texts are null. A pair moves together across splits. `local_image` is relative to the data root. Distinct claims sharing a source image path must stay in the same split; construction and training preparation reject cross-split path overlap. Copies of an image under different paths still require a separate duplicate-image audit of the benchmark. `cot_response` and `teacher_model` are optional training annotations; when no teacher trace exists, the SFT converter uses a short target derived from the reviewed labels.

`gt_entity_fine` describes the **actual visible replacement**. It must not contain only the intended edit instruction or a before/after pair. `gt_why_contradicts` describes the world-knowledge contradiction. `gt_entity_canonical` and integer `year` are additionally required when constructing entity/temporal splits.

## Type labels

| Release label | Legacy construction label |
| --- | --- |
| clothing | clothing |
| flag | flag |
| gesture | social_behavior |
| signage | text_language |
| architecture | architecture |
| infrastructure | infrastructure |
| technology | technology |
| branding | ads_anachronism |
| environment | environmental |

Legacy names are accepted for source records and normalized by `src/records.py`. Model outputs must use the release labels. The legacy `<grounding>` and `<knowledge>` fields correspond to release `<visual>` and `<explanation>` fields; new training targets are rebuilt from the reviewed reference fields.

See the main README for [training](../README.md#training) and [evaluation](../README.md#inference-and-evaluation).
