# Third-party notices

The root Apache-2.0 license covers MIC's original code. It does not replace upstream notices.

| Component | Upstream | License / notice in this repository |
| --- | --- | --- |
| LlamaFactory | https://github.com/hiyouga/LlamaFactory | [Apache-2.0](LlamaFactory/LICENSE) |
| verl | https://github.com/volcengine/verl | [Apache-2.0](training/verl/LICENSE), [Notice.txt](training/verl/Notice.txt) |

These are bundled research snapshots: LlamaFactory `0.9.5.dev0` and verl `0.8.0.dev0`. Existing per-file copyright headers remain in place. MIC-specific changes include dataset/reward integration, multimodal prompt processing, and portable release entry points.

The construction and preparation utilities were adapted from the companion `ai_image_inconsistency` research project. The paper's historical labels are mapped to the MIC release schema in `src/data.py`.

TARA data and source images are downloaded separately from the [authors' release](https://github.com/zeyofu/TARA). Qwen model weights, embedding weights, and hosted image services are separate resources with their own terms. No image or model license is implied by the MIC code license.

The repository layout and project metadata are adapted from the [UKP project template](https://github.com/UKPLab/ukp-project-template). The UKP copyright notice is retained in [NOTICE.txt](NOTICE.txt). The bundled verl MIC reward import now points to `src.training.reward`.
