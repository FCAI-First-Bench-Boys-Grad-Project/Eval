import os
import json
import signal
from typing import Optional, Dict, Any, List
from tqdm.auto import tqdm
from math import ceil

from html_eval.configs.experiment_config import ExperimentConfig
from html_eval.pipelines.base_pipeline import BasePipeline
from html_eval.eval.evaluator import Evaluator
from html_eval.html_datasets.base_html_dataset import BaseHTMLDataset
from html_eval.util.seed_util import set_seed
from html_eval.util.file_util import atomic_write
from html_eval.core.tracker import BaseTracker, get_tracker


class Experiment:
    """
    Orchestrates an experiment:
    - Loads dataset, pipeline, evaluator
    - Iterates through data in batches
    - Generates predictions
    - Evaluates results
    - Logs metrics to chosen tracker (MLflow/W&B/etc.)
    - Saves progress via checkpoints for resumability
    """

    def __init__(
        self,
        config: ExperimentConfig,
        tracker_backend: str = "wandb",
        resume: bool = False,
        project_name: Optional[str] = None,
    ):
        self._config = config
        set_seed(self._config.seed)

        # core components
        self.data: BaseHTMLDataset = config.dataset_config.create_dataset()
        self.pipeline: BasePipeline = config.pipeline_config.create_pipeline()
        self.evaluator: Evaluator = Evaluator(
            evaluation_metrics=self._config.dataset_config.evaluation_metrics
        )
        self.tracker: BaseTracker = get_tracker(backend=tracker_backend, 
                                                project=project_name,
                                                experiment_name=self._config.experiment_name,
                                                config=self._config.to_dict())

        # checkpoint state
        self._output_dir = self._config.output_dir

        self._results_file = os.path.join(self._output_dir, "results.json")
        self._metric_dir = os.path.join(self._output_dir, "metric")
        self._checkpoint_file = os.path.join(
            self._output_dir, "checkpoint.json"
        )
        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(self._metric_dir, exist_ok=True)
        
        self._progress = {"experiment_config":self._config,"last_batch": -1, "predictions": []}
        
        if resume:
            self._load_checkpoint()
        else:
            if os.path.exists(self._checkpoint_file):
                print(f"[Experiment] Ignoring existing checkpoint: {self._checkpoint_file}")

        # connect references
        self.pipeline.set_experiment(self)
        self.data.set_experiment(self)
        self.evaluator.set_experiment(self)

        # graceful shutdown
        self._init_signals()

    # ------------------- Lifecycle Helpers -------------------
    def _init_signals(self) -> None:
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)

    def _graceful_shutdown(self, signum, frame) -> None:
        print(f"\n[Experiment] Received signal {signum}, saving checkpoint...")
        self._save_checkpoint()
        self.tracker.finish()
        exit(0)

    def _save_results(self,results) -> None:
        if os.path.exists(self._results_file):
            print(f"[Experiment] Overwriting existing results file: {self._results_file}")
        with open(self._results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Experiment] Saved results to {self._results_file}")

    def _save_sample_eval(self, sample_eval , metric_name) -> None:
        os.makedirs(os.path.join(self._metric_dir,metric_name), exist_ok=True)
        file_name = os.path.join(self._metric_dir,metric_name,"sample_eval.json")
        if os.path.exists(file_name):
            print(f"[Experiment] Overwriting existing sample eval file: {file_name}")
        with open(file_name, "w") as f:
            json.dump(sample_eval, f, indent=2)
        print(f"[Experiment] Saved sample evaluation to {file_name}")

    def _save_config(self) -> None:
        config_file = os.path.join(self._output_dir, "experiment_config.json")
        if os.path.exists(config_file):
            print(f"[Experiment] Overwriting existing config file: {config_file}")
        with open(config_file, "w") as f:
            print("FUCKKKKK ",self._config.to_dict())
            json.dump(self._config.to_dict(), f, indent=2)
        print(f"[Experiment] Saved experiment config to {config_file}")
    # ------------------- Checkpointing -------------------
    def _save_checkpoint(self) -> None:
        atomic_write(self._checkpoint_file, self._progress)

    def _load_checkpoint(self) -> None:
        if os.path.exists(self._checkpoint_file):
            with open(self._checkpoint_file, "r") as f:
                self._progress = json.load(f)
            print(f"[Experiment] Resuming from batch {self._progress['last_batch'] + 1}")

    # ------------------- Main Loop -------------------
    #FIXME: batch size isn't connected
    def run(self) -> Dict[str, Any]:
        set_seed(self._config.seed)
        batch_size = self._config.dataset_config.batch_size
        shuffle = self._config.dataset_config.shuffle

        predictions: List[Dict[str, Any]] = self._progress["predictions"]

        if batch_size and hasattr(self.data, "batch_iterator"):
            iterator = self.data.batch_iterator(batch_size=batch_size, shuffle=shuffle)
            total = ceil(len(self.data) / batch_size)
        else:
            iterator = iter(self.data)
            total = len(self.data)

        try:
            for batch_idx, batch in enumerate(
                tqdm(iterator, desc="Running Experiment", unit="batch", total=total)
            ):
                if batch_idx <= self._progress["last_batch"]:
                    continue  # already processed, skip

                try:
                    pred = self.pipeline.extract(batch)
                    predictions.extend(pred)

                    # update checkpoint state
                    self._progress.update(
                        last_batch=batch_idx,
                        predictions=predictions,
                    )
                    self._save_checkpoint()
                except Exception as e:
                    print(f"[Experiment] Error on batch {batch_idx}: {e}")
                    continue

            print("[Experiment] Finished. Evaluating...")
            results = self.evaluator.evaluate(predictions)

            # Saving results
            self._save_results(results)
            metrics = self.evaluator.get_metrics()
            for metric in metrics:
                self._save_sample_eval(metric._sample_eval, metric.name())
                metric._sample_eval
            self._save_config()
            


            self.tracker.log_metrics(results, step=self._progress["last_batch"] + 1)
            # self.tracker.log_artifact()


            return {"predictions": predictions, "results": results}

        finally:
            # always finish tracker, even on crash
            self.tracker.finish()
