from abc import ABC, abstractmethod
from typing import Dict, Any
import polars as pl
class BasePipeline(ABC):
    def __init__(self, **kwargs):
        pass  # common setup if needed

    @abstractmethod
    def extract(self, batch: pl.DataFrame) -> Dict[str, Any]:
        """
        Extract information from a batch of content.
        """
        pass

