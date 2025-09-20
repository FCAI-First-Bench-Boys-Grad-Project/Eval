from typing import Optional, Dict, Any
import polars as pl
from eval.methods.base import BasePipeline
from eval import Evaluator
from eval.html_datasets.base import BaseHTMLDataset
from eval.util.seed import set_seed
from tqdm.auto import tqdm
from math import ceil




class Experiment:
    def __init__(
        self,
        config: Dict[str, Any],  # configuration dictionary
    ):
        self.config = config




        self.data      = data
        self.pipeline  = pipeline
        self.evaluator = evaluator
        
        # Connecting the Modules to the Experiment
        self.pipeline.set_experiment(self)
        self.evaluator.set_experiment(self)
        # self.data.set_experiment(self)
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
        ground_truths = []

        if batch_size is not None and hasattr(self.data, 'batch_iterator'):
            iterator = self.data.batch_iterator(batch_size=batch_size, shuffle=shuffle)
        else:
            iterator = iter(self.data)

        # tqdm wrapper — leave total=None if you don’t know dataset length
        for batch in tqdm(iterator, desc="Running Experiment", unit="batch", total=ceil(getattr(self.data, "__len__", lambda: None)()/ batch_size if batch_size else 1)):
            if isinstance(batch, list):
                html, query, gt = zip(*batch)
            else:
                html, query, gt = batch

            # Turn html and query into a polars DataFrame
            batch_df = pl.DataFrame({
                'html': html,
                'query': query
            })

            # FIXME: Robustness against failure and failed indexes
            # print(f"Batch Shape {batch_df.shape}")
            pred = self.pipeline.extract(batch_df)
            
            predictions.extend(pred)
            ground_truths.extend(gt)
            

        # print("Predictions")
        # print(predictions)
        # print("GT")
        # print(ground_truths)
        
        # TODO MLFlow
        print("Please Tell me this is not where the problem is")
        # print(f"Predictions Length: {pl.Series(predictions)}")
        # print(f"Ground Truths Length: {pl.Series(ground_truths)}")
        

        ground_truths = pl.Series(ground_truths)
        results  = self.evaluator.evaluate(pl.Series(predictions,dtype=pl.Object), pl.Series(ground_truths))
        return predictions , ground_truths , results
