from dataclasses import dataclass , field
from typing import Optional
from os import cpu_count
from html_eval.configs.llm_client_config import LLMClientConfig

@dataclass
class BasePipelineConfig:

    def create_pipeline(self):
        raise NotImplementedError("Subclasses should implement this method.")

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
        self.fetch_workers = self.fetch_workers if self.fetch_workers is not None else min(32, max(4, (cpu_count() or 2) * 2))
        default_cpu = max(1, (cpu_count() or 2) - 1)
        self.cpu_workers = self.cpu_workers if self.cpu_workers is not None else default_cpu

@dataclass
class RerankerExtractorConfig:
    llm_config: LLMClientConfig
    classification_prompt_template: str
    generation_prompt_template: str

    reranker_huggingface_model: str = "abdo-Mansour/Qwen3-Reranker-0.6B-HTML"
    reranker_max_prompt_length: int = 8192 # this is the max length of the prompt (question + context)
    reranker_max_total_length: int = 2048  # this the max length of the prompt + response
    reranker_default_top_k: int = None
    reranker_tensor_parallel_size: int = None
    reranker_quantization: str = "bitsandbytes"
    reranker_gpu_memory_utilization: float = 0.7
    reranker_enable_prefix_caching: bool = True
    reranker_classification_threshold: float = 0.5
    

@dataclass
class RerankerPostprocessorConfig:
    exact_extraction: bool = False

@dataclass
class RerankerPipelineConfig(BasePipelineConfig):
    '''
    Okay here are the things I should care for this pipeline:
    - name
    - preprocessor config
    - extractor config
    - postprocessor config
        - llm config
    '''

    preprocessor_config: RerankerPreprocessorConfig
    extractor_config: RerankerExtractorConfig
    postprocessor_config: RerankerPostprocessorConfig
    name: str = "reranker"
    def create_pipeline(self):
        from html_eval.pipelines.reranker.pipeline import RerankerPipeline
        return RerankerPipeline(self)


############### END RERANKER PIPELINE CONFIG #########