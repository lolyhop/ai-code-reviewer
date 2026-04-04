import json
import logging
import typing as tp

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ai_code_reviewer.models.config import GenerationConfig, ModelConfig
from src.ai_code_reviewer.models.schema import PredictedIssue, ReviewPrediction

logger = logging.getLogger(__name__)

TORCH_DTYPE_MAP: tp.Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ReviewModel:
    """Wrapper around Qwen3 for code-review inference."""

    def __init__(self, model_config: tp.Optional[ModelConfig] = None) -> None:
        self.config = model_config or ModelConfig()
        self._tokenizer: tp.Optional[AutoTokenizer] = None
        self._model: tp.Optional[AutoModelForCausalLM] = None

    def load(self) -> None:
        """Load tokenizer and model into memory."""
        if self._model is not None:
            return

        dtype = TORCH_DTYPE_MAP.get(self.config.torch_dtype, torch.bfloat16)

        logger.info("Loading tokenizer %s ...", self.config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        load_kwargs: tp.Dict[str, tp.Any] = {
            "torch_dtype": dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )

        logger.info("Loading model %s ...", self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )
        self._model.eval()
        logger.info("Model loaded successfully.")

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            raise RuntimeError("Call .load() before using the model.")
        return self._tokenizer

    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            raise RuntimeError("Call .load() before using the model.")
        return self._model

    def _build_chat_input(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate_raw(
        self,
        prompt: str,
        gen_config: tp.Optional[GenerationConfig] = None,
    ) -> str:
        """Run generation and return the raw decoded string."""
        self.load()
        cfg = gen_config or GenerationConfig()

        chat_text = self._build_chat_input(prompt)

        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature if cfg.do_sample else None,
                top_p=cfg.top_p if cfg.do_sample else None,
                do_sample=cfg.do_sample,
                repetition_penalty=cfg.repetition_penalty,
            )

        new_tokens = output_ids[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def predict(
        self,
        prompt: str,
        gen_config: tp.Optional[GenerationConfig] = None,
    ) -> ReviewPrediction:
        """Generate and parse into ReviewPrediction."""
        raw = self.generate_raw(prompt, gen_config)
        return self._parse_response(raw)

    def predict_batch(
        self,
        prompts: tp.List[str],
        gen_config: tp.Optional[GenerationConfig] = None,
    ) -> tp.List[ReviewPrediction]:
        """Run predict for each prompt sequentially."""
        return [self.predict(p, gen_config) for p in prompts]

    @staticmethod
    def _parse_response(raw: str) -> ReviewPrediction:
        """Best-effort JSON parse of model output."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse model output as JSON: %s", text[:200]
                    )
                    return ReviewPrediction()
            else:
                logger.warning("No JSON object found in model output: %s", text[:200])
                return ReviewPrediction()

        issues = []
        for item in data.get("issues", []):
            lr = item.get("line_range", {})
            issues.append(
                PredictedIssue(
                    line_start=lr.get("start"),
                    line_end=lr.get("end"),
                    comment=item.get("comment", ""),
                )
            )
        return ReviewPrediction(issues=issues)
