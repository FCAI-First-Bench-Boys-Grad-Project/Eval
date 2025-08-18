from typing import Optional, Dict, Any
import polars as pl
# from tqdm import tqdm
from tqdm.notebook import tqdm
from math import ceil
class Experiment:
    def __init__(
        self, 
        data,            # dataset instance
        pipeline,        # model or processing pipeline instance
        evaluator,       # evaluator instance
    ):
        self.data = data
        self.pipeline = pipeline
        self.evaluator = evaluator

    def run(self, batch_size: Optional[int] = None, shuffle: bool = False) -> Dict[str, Any]:
        """
        Run the experiment end-to-end:
        - Iterate over data (optionally batched)
        - Pass data through pipeline to get predictions
        - Evaluate predictions against ground truths
        """

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
            # print(f"Batch Shape {batch_df.shape}")
            pred = self.pipeline.extract(batch_df)
            # print(pred)
            predictions.append(pred)
            ground_truths.append(gt)

        # print("Predictions")
        # print(predictions)
        # print("GT")
        # print(ground_truths)
        
        # results = self.evaluator.compute_metrics(predictions, ground_truths)
        return predictions , ground_truths
