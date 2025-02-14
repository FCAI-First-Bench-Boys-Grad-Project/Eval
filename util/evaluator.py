from abc import ABC, abstractmethod
from typing import List, Dict, Any
from numpy.typing import NDArray
import numpy as np

import json
from jsonschema import validate, ValidationError


class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def calculate(self,ground_truth: NDArray[Any], predictions: NDArray[Any]) -> float:
        """
        Calculate the metric value.
        
        Args:
            ground_truth: Ground truth output
            predictions: Predicted output
            
        Returns:
            float: Calculated metric value
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""
        pass

class RecallJson(Metric):
    """The proportion of all relevant key information that is successfully extracted. Performing Macro averaging."""
    
    def calculate(self, ground_truth: NDArray[Any], predictions: NDArray[Any]) -> float:
        '''
        The input should be a numpy array of dictionaries.
        '''

        true_positive = 0
        # Loop through each dictionary in the ground truth and check if it is in the predictions
        for i in range(len(ground_truth)):
            sample_true_positive = 0
            for key , value in ground_truth[i].items():
                if key in predictions[i].keys() and value == predictions[i][key]:
                    sample_true_positive += 1
            true_positive += sample_true_positive / len(ground_truth[i])

        return true_positive / len(ground_truth)

                    
    
    @property
    def name(self) -> str:
        return "recall"
    
class PrecisionJson(Metric):
    """The proportion of all extracted key information that is relevant. Performing Macro averaging."""
    
    def calculate(self, ground_truth: NDArray[Any], predictions: NDArray[Any]) -> float:
        '''
        The input should be a numpy array of dictionaries.
        '''
        true_positive = 0
        for i in range(len(ground_truth)):
            sample_true_positive = 0
            for key , value in ground_truth[i].items():
                if key in predictions[i].keys() and value == predictions[i][key]:
                    sample_true_positive += 1
            true_positive += sample_true_positive / len(predictions[i])
        return true_positive / len(predictions)
                
    @property
    def name(self) -> str:
        return "precision"

class F1Json(Metric):
    """The harmonic mean of precision and recall. Performing Macro averaging."""
    
    def calculate(self, ground_truth: NDArray[Any], predictions: NDArray[Any]) -> float:
        '''
        The input should be a numpy array of dictionaries.
        '''
        precision = PrecisionJson().calculate(ground_truth, predictions)
        recall = RecallJson().calculate(ground_truth, predictions)
        return 2 * (precision * recall) / (precision + recall)
    
    @property
    def name(self) -> str:
        return "f1"

class Evaluator:
    """Class for evaluating predictions using multiple metrics."""
    
    def __init__(self, metrics: List[Metric]):
        """
        Initialize the evaluator with a list of metrics.
        
        Args:
            metrics: List of Metric objects to use for evaluation
        """
        self.metrics = metrics
    
    def evaluate(self, ground_truth: NDArray[Any], predictions: NDArray[Any]) -> Dict[str, float]:
        """
        Evaluate predictions using all metrics.
        
        Args:
            ground_truth: Ground truth output
            predictions: Predicted output
            
        Returns:
            Dict mapping metric names to their calculated values
        """
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.calculate(ground_truth, predictions)
        return results