from abc import ABC, abstractmethod
from html_eval.configs.pipeline_config import BasePipelineConfig
from typing import List , TYPE_CHECKING
from html_eval.core.types import Sample , SamplePrediction

if TYPE_CHECKING:
    from html_eval.core.experiment import Experiment

class BasePipeline(ABC):
    def __init__(self,config: BasePipelineConfig):
        self.config : BasePipelineConfig = config
        self.experiment : "Experiment" = None
    
    def set_experiment(self, experiment) -> None:
        """
        Set the experiment instance for logging or other purposes.
        """
        self.experiment = experiment
        
    @abstractmethod
    def extract(self, batch: List[Sample]) -> List[SamplePrediction]:
        """
        Extract information from a batch of content.
        """
        pass

