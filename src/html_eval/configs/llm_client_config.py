from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from html_eval.core.llm import LLMClient, NvidiaLLMClient, VLLMClient


@dataclass
class LLMClientConfig:
    llm_source: str = "nvidia"
    model_name: str = "google/gemma-3n-e2b-it"
    api_key: Optional[str] = None  # If None, will look for env var
    temperature: float = 0.0
    top_p: float = 0.7
    max_tokens: int = 8192

    def create_llm_client(self) -> LLMClient:
        config = {
            "model_name": self.model_name,
            "api_key": self.api_key,
            "generation_config": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            },
        }
        if self.llm_source == "nvidia":
            return NvidiaLLMClient(config=config)
        elif self.llm_source == "vllm":
            return VLLMClient(config=config)
        else:
            raise ValueError(f"Unsupported llm_source: {self.llm_source}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dict."""
        return asdict(self)
