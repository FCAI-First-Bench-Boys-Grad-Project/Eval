from typing import Optional, Dict, Any

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

        for batch in iterator:
            # batch can be a list of tuples or single tuple depending on 
            if batch_size is not None:
                # batch mode
                batch_preds = [self.pipeline.extract(html, query) for html, query, _ in batch]
                batch_gts = [gt for _, _, gt in batch]
                predictions.extend(batch_preds)
                ground_truths.extend(batch_gts)
            else:
                # single sample mode
                html, query, gt = batch
                pred = self.pipeline.extract(html, query)
                print(f"Processed: {pred}")
                print(f"Ground Truth: {gt}")
                predictions.append(pred)
                ground_truths.append(gt)

        print("Predictions")
        print(predictions)
        print("GT")
        print(ground_truths)
        # Compute metrics using the evaluator
        # results = self.evaluator.compute_metrics(predictions, ground_truths)
        # return results
