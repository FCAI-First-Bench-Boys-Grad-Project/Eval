from dataclasses import dataclass
from html_eval.configs.pipeline_config import BasePipelineConfig
from html_eval.configs.dataset_config import BaseDatasetConfig

@dataclass
class ExperimentConfig:
    experiment_name: str
    pipeline_config: BasePipelineConfig
    dataset_config: BaseDatasetConfig
    seed: int = 42

    def __post_init__(self):
        if self.experiment_name.strip() == "":
            raise ValueError("experiment_name cannot be empty")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.pipeline_config, BasePipelineConfig):
            raise ValueError("pipeline_config must be an instance of BasePipelineConfig")
        if not isinstance(self.dataset_config, BaseDatasetConfig):
            raise ValueError("dataset_config must be an instance of BaseDatasetConfig")
        

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "pipeline_config": self.pipeline_config,
            "dataset_config": self.dataset_config,
        }
    


    
