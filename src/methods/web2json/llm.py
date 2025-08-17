from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import os
import random
import time
from typing import List, Iterable, Optional, Any, Callable

from openai import OpenAI
from openai import RateLimitError


def retry_on_ratelimit(max_retries=5, base_delay=1.0, max_delay=10.0):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    sleep = min(max_delay, delay) + random.uniform(0, delay)
                    time.sleep(sleep)
                    delay *= 2
        return wrapped
    return deco


class LLMClient(ABC):
    """
    Abstract base class for calling LLM APIs.
    Provides a default call_batch implementation that calls call_api in parallel.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def call_api(self, prompt: str, **kwargs) -> str:
        """
        Call the underlying LLM API with a single prompt.
        Must be implemented by subclasses.

        Args:
            prompt: prompt string
            kwargs: vendor-specific options

        Returns:
            response string
        """
        raise NotImplementedError

    def call_batch(
        self,
        prompts: Iterable[str],
        max_workers: int = 8,
        chunk_size: Optional[int] = None,
        raise_on_error: bool = False,
        per_result_callback: Optional[Callable[[int, Optional[str], Optional[Exception]], Any]] = None,
        **call_api_kwargs,
    ) -> List[Optional[str]]:
        """
        Default batch implementation: runs call_api in parallel with a ThreadPoolExecutor.
        Preserves order of input prompts in returned list.

        Args:
            prompts: iterable of prompt strings
            max_workers: max parallel workers
            chunk_size: if set, splits the prompts into chunks of this size and processes sequentially.
                        Useful to limit concurrency or rate.
            raise_on_error: if True, re-raise the exception when any prompt fails after retries.
            per_result_callback: optional function called as callback(idx, result, exception) for each finished prompt.
            call_api_kwargs: forwarded to call_api.

        Returns:
            list of responses (or None for failed items if raise_on_error is False)
        """
        prompts = list(prompts)
        results: List[Optional[str]] = [None] * len(prompts)

        def _submit_range(start_idx: int, end_idx: int):
            # submit jobs for a slice [start_idx, end_idx)
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i in range(start_idx, end_idx):
                    fut = ex.submit(self.call_api, prompts[i], **call_api_kwargs)
                    futures[fut] = i
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        res = fut.result()
                        results[idx] = res
                        if per_result_callback:
                            per_result_callback(idx, res, None)
                    except Exception as exc:
                        # If caller wants to raise, do so; otherwise set None and continue
                        if per_result_callback:
                            per_result_callback(idx, None, exc)
                        if raise_on_error:
                            raise
                        results[idx] = None

        if chunk_size is None or chunk_size <= 0:
            # one-shot submit all prompts (bounded by max_workers in each executor)
            _submit_range(0, len(prompts))
        else:
            # process chunks sequentially to throttle
            for start in range(0, len(prompts), chunk_size):
                end = min(start + chunk_size, len(prompts))
                _submit_range(start, end)

        return results


class NvidiaLLMClient(LLMClient):
    """
    Concrete implementation of LLMClient for the NVIDIA API (non-streaming).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        api_key = config.get("api_key") or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for NVIDIA must be provided in config['api_key'] or NVIDIA_API_KEY env var."
            )

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model_name = config.get("model_name", "google/gemma-3-1b-it")

        gen_conf = config.get("generation_config", {})
        self.temperature = gen_conf.get("temperature", 0)
        self.top_p = gen_conf.get("top_p", 0.7)
        self.max_tokens = gen_conf.get("max_tokens", 8192)

    def set_model(self, model_name: str):
        self.model_name = model_name

    @retry_on_ratelimit(max_retries=20, base_delay=0.5, max_delay=5.0)
    def call_api(self, prompt: str, **kwargs) -> str:
        """
        Single prompt call (non-streaming).
        kwargs are forwarded to the underlying API call if needed.
        """
        # print("prompt:", prompt)  # keep optional for debugging
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"thinking": True}},
            # any additional kwargs can be merged here if needed
        )
        return response.choices[0].message.content

    # Optionally override call_batch if the vendor supports true batched calls.
    # For now, we inherit the default implementation from LLMClient.
