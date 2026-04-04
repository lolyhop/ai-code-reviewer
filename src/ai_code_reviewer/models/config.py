from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    max_input_length: int = 4096


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.0