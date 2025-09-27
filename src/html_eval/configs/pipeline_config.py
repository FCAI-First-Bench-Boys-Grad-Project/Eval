import dataclasses
from dataclasses import dataclass, asdict, field
from typing import Optional, Any, Dict
from os import cpu_count
from html_eval.configs.llm_client_config import LLMClientConfig


@dataclass
class BasePipelineConfig:
    def create_pipeline(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Recursively convert dataclass instance to a dictionary,
        properly handling nested dataclasses and objects with a `to_dict` method.
        """
        result = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "to_dict"):
                result[f.name] = value.to_dict()
            elif isinstance(value, list):
                result[f.name] = [item.to_dict() if hasattr(item, "to_dict") else item for item in value]
            elif isinstance(value, dict):
                result[f.name] = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in value.items()}
            else:
                result[f.name] = value
        return result


############# RERANKER PIPELINE CONFIG #############
@dataclass
class RerankerPreprocessorConfig:
    fetch_workers: Optional[int] = None
    cpu_workers: Optional[int] = None

    extra_remove_tags: list = field(default_factory=lambda: ["header", "footer"])
    strip_attrs: bool = True
    strip_links: bool = True
    keep_tags: bool = True
    use_clean_rag: bool = True
    use_clean_chunker: bool = True

    chunk_size: int = 500
    attr_cutoff_len: int = 5

    def __post_init__(self):
        self.fetch_workers = (
            self.fetch_workers
            if self.fetch_workers is not None
            else min(32, max(4, (cpu_count() or 2) * 2))
        )
        default_cpu = max(1, (cpu_count() or 2) - 1)
        self.cpu_workers = (
            self.cpu_workers if self.cpu_workers is not None else default_cpu
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RerankerExtractorConfig:
    llm_config: LLMClientConfig
    classification_prompt_template: str
    generation_prompt_template: str

    reranker_huggingface_model: str = "abdo-Mansour/Qwen3-Reranker-0.6B-HTML"
    reranker_max_prompt_length: int = 8192
    reranker_max_total_length: int = 2048
    reranker_default_top_k: Optional[int] = None
    reranker_tensor_parallel_size: Optional[int] = None
    reranker_quantization: str = "bitsandbytes"
    reranker_gpu_memory_utilization: float = 0.7
    reranker_enable_prefix_caching: bool = True
    reranker_classification_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        # This implementation correctly handles the nested llm_config
        return {
            **asdict(self),
            "llm_config": (
                self.llm_config.to_dict()
                if hasattr(self.llm_config, "to_dict")
                else str(self.llm_config)
            ),
        }


@dataclass
class RerankerPostprocessorConfig:
    exact_extraction: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RerankerPipelineConfig(BasePipelineConfig):
    """
    Pipeline config for the reranker pipeline.
    Includes preprocessor, extractor, postprocessor configs and a pipeline name.
    """

    preprocessor_config: RerankerPreprocessorConfig
    extractor_config: RerankerExtractorConfig
    postprocessor_config: RerankerPostprocessorConfig
    name: str = "reranker"

    def create_pipeline(self):
        from html_eval.pipelines.reranker.pipeline import RerankerPipeline

        return RerankerPipeline(self)



############### END RERANKER PIPELINE CONFIG #########