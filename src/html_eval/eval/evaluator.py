from __future__ import annotations
from typing import Any, Dict, List, Optional
from html_eval.core.types import SamplePrediction
from html_eval.eval.metric import Metric, METRICS_REGISTRY
from html_eval.eval.matcher import MatcherConfig
from html_eval.util.eval_util import is_not_null

class Evaluator:

    def __init__(self, evaluation_metrics: List[str],
                 matcher_cfg: Optional[MatcherConfig] = None):
        
        self.evaluation_metrics = evaluation_metrics
        self.matcher_cfg = matcher_cfg or MatcherConfig()

        # instantiate metric objects
        self.metrics = [] 
        for name in evaluation_metrics:
            if name not in METRICS_REGISTRY:
                raise KeyError(f"Unknown metric {name}")
            self.metrics.append(METRICS_REGISTRY[name](self))
    
    def get_metrics(self) -> List[Metric]:
        """
        Returns the list of instantiated metrics.
        """
        return self.metrics
    
    def set_experiment(self, experiment: Any):
        """
        Set the experiment context for the evaluator.
        This can be used to log results, etc.
        """
        self.experiment = experiment

    def evaluate(self, predictions: List[SamplePrediction]) -> Dict[str, Any]:
        # if not is_not_null(predictions):
        #     raise ValueError("Both pred and gt must contain non-null values")
        
        print("I swear if the problem is from here I will fruck you up")
        print("Pred: ", predictions)
        
        results: Dict[str, Any] = {}
        for metric in self.metrics:
            print("Calculating with Pred: ", predictions)
            res = metric.calculate(predictions)
            results[metric.name()] = res
        
        return results

