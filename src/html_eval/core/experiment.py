from typing import Optional, Dict, Any
import polars as pl
from html_eval.configs.experiment_config import ExperimentConfig 
from html_eval.pipelines.base_pipeline import BasePipeline
from html_eval.eval.evaluator import Evaluator
from html_eval.html_datasets.base_html_dataset import BaseHTMLDataset
from html_eval.util.seed_util import set_seed
from tqdm.auto import tqdm
from math import ceil




class Experiment:
    def __init__(self,config: ExperimentConfig):
        self._config: ExperimentConfig = config
        
        set_seed(self._config.seed)
        self.data: BaseHTMLDataset = config.dataset_config.create_dataset()
        self.pipeline: BasePipeline = config.pipeline_config.create_pipeline()
        self.evaluator: Evaluator = Evaluator(evaluation_metrics=self._config.dataset_config.evaluation_metrics)
        
        # Connecting the Modules to the Experiment
        self.pipeline.set_experiment(self)
        self.data.set_experiment(self)
        self.evaluator.set_experiment(self)
        # TODO MLflow
    
    # @app.task
    def run(self, batch_size: Optional[int] = None, shuffle: bool = False) -> Dict[str, Any]:
        """
        Run the experiment end-to-end:
        - Iterate over data (optionally batched)
        - Pass data through pipeline to get predictions
        - Evaluate predictions against ground truths
        """
        set_seed(42)  # For reproducibility

        predictions = []

        if batch_size is not None and hasattr(self.data, 'batch_iterator'):
            iterator = self.data.batch_iterator(batch_size=batch_size, shuffle=shuffle)
        else:
            iterator = iter(self.data)

        # tqdm wrapper — leave total=None if you don’t know dataset length
        for batch in tqdm(iterator, desc="Running Experiment", unit="batch", total=ceil(getattr(self.data, "__len__", lambda: None)()/ batch_size if batch_size else 1)):
            
            pred = self.pipeline.extract(batch)
            
            predictions.extend(pred)
            

        print("Predictions")
        print(predictions)
        
        
        # TODO MLFlow

        results  = self.evaluator.evaluate(predictions)
        return predictions , results
