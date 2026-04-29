import logging
import os
import time
import typing as tp

import openai
import urllib3

from ai_code_reviewer.utils import parse_json_response

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible client."""

    def __init__(
        self,
        api_key: tp.Optional[str] = None,
        folder_id: tp.Optional[str] = None,
        model: str = "deepseek-v32/latest",
        base_url: str = "https://ai.api.cloud.yandex.net/v1",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("YANDEX_CLOUD_API_KEY", "")
        self.folder_id = folder_id or os.environ.get("YANDEX_CLOUD_FOLDER", "")
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            project=self.folder_id,
        )

    @property
    def model_uri(self) -> str:
        return f"gpt://{self.folder_id}/{self.model}"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> str:
        """Send a prompt and return the raw response text."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.responses.create(
                    model=self.model_uri,
                    temperature=temperature,
                    instructions=system_prompt,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": "none"},
                )
                return response.output_text
            except Exception as exc:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise
        return ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> tp.Dict[str, tp.Any]:
        """Generate a response and parse it as JSON."""
        raw = self.generate(prompt, system_prompt, max_tokens, temperature)
        return parse_json_response(raw)
