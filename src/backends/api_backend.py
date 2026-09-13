"""API backend — unified sync interface over closed-model providers.

Providers currently supported:
  openai    — GPT-5.4-mini, GPT-4o-mini, etc. (env: OPENAI_API_KEY)

Interface mirrors VLLMBackend: load() / run(image_path, prompt) -> str / unload().
No batching here — upstream run_infer.py can parallelize via ThreadPoolExecutor.
"""

import base64
import os
import time
from pathlib import Path


from src.environment import load_environment

load_environment()


def _image_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(ext, "jpeg")


def _image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------- OpenAI ----------

class _OpenAIProvider:
    def __init__(self, api_model: str, temperature: float = 0.0):
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set (checked env + .env)")
        self.client = OpenAI()
        self.api_model = api_model
        self.temperature = temperature

    def run(self, image_path: str, prompt: str) -> str:
        media = _image_media_type(image_path)
        b64 = _image_to_base64(image_path)
        resp = self.client.chat.completions.create(
            model=self.api_model,
            **({} if self.api_model.startswith(("gpt-5", "o1", "o3", "o4")) else {"temperature": self.temperature}),
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{media};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return resp.choices[0].message.content or ""


_PROVIDERS = {
    "openai": _OpenAIProvider,
}


class APIBackend:
    """Unified API backend with lazy provider init + retry."""

    def __init__(
        self,
        provider: str,
        api_model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        if provider not in _PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Expected one of {list(_PROVIDERS)}")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.provider = provider
        self.api_model = api_model
        self.temperature = temperature
        self.max_retries = max_retries
        self._impl = None

    # Match VLLMBackend interface
    @property
    def model_name(self) -> str:
        return f"{self.provider}:{self.api_model}"

    def load(self):
        self._impl = _PROVIDERS[self.provider](self.api_model, self.temperature)
        print(f"[api] provider={self.provider} model={self.api_model} ready")

    def unload(self):
        if self._impl is not None:
            self._impl.client.close()
        self._impl = None

    def run(self, image_path: str, prompt: str) -> str:
        if self._impl is None:
            self.load()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                return self._impl.run(image_path, prompt)
            except Exception as e:
                last_err = e
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2 ** attempt, 15))
        raise RuntimeError(f"{self.provider}:{self.api_model} failed after {self.max_retries} retries: {last_err}")
