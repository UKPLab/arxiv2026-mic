# Data construction

MIC-Bench is built from authentic TARA image–claim pairs. The pipeline keeps each claim unchanged and introduces one contextual inconsistency into an edited counterpart.

1. **Prepare source data.** Download TARA's New York Times metadata and source images, retaining images with width and height of at least 1,024 pixels.
2. **Screen and verify.** GPT-4o-mini screens editing suitability, and GPT-5.5 verifies which contextual elements are clearly visible.
3. **Generate edits.** Assign one inconsistency type per image, use GPT-5.5 to propose the modification, and generate the edited image with GPT-Image-1.5.
4. **Review and annotate.** Human reviewers check the generated image and finalize its inconsistency type, visual evidence, and factual explanation. Test candidates require approval from two independent annotators.
5. **Create splits.** Keep each original/edited pair together and construct training, validation, ID, and OOD splits using temporal and replacement-entity separation.
6. **Prepare training data.** Generate teacher rationales with GPT-5.5, revise them through human review, and convert the reviewed records into SFT and GRPO formats.

The resulting dataset contains paired original and edited images with structured annotations. Images are stored under `data/TARA/`, and split manifests under `data/splits/`. Re-running generation produces new images; the exact MIC-Bench download is pending.

The construction prompts in [prompts.py](prompts.py) follow the paper's Prompts appendix. See the [data schema](../data/README.md) for file formats, the [training instructions](../README.md#training) for conversion commands, and `python -m data_construction --help` for available stages.
