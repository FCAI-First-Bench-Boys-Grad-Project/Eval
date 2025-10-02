import os
import json
import signal
import gc
from typing import Optional, Dict, Any, List, Iterable
from tqdm.auto import tqdm
from math import ceil
import ast
# optional imports
try:
    import torch
except Exception:
    torch = None

from html_eval.configs.experiment_config import ExperimentConfig
from html_eval.pipelines.base_pipeline import BasePipeline
from html_eval.eval.evaluator import Evaluator
from html_eval.html_datasets.base_html_dataset import BaseHTMLDataset
from html_eval.core.types import SamplePrediction 
from html_eval.util.seed_util import set_seed
from html_eval.util.file_util import atomic_write
from html_eval.core.tracker import BaseTracker, get_tracker


class Experiment:
    """
    Streaming-friendly Experiment:
    - writes batch predictions to NDJSON (predictions.ndjson)
    - keeps checkpoint tiny (only last_batch + metadata)
    - optionally updates evaluator incrementally if evaluator.update exists
    """

    def __init__(
        self,
        config: ExperimentConfig,
        tracker_backend: str = "wandb",
        resume: bool = False,
        project_name: Optional[str] = None,
        checkpoint_every: int = 10,
        flush_every: int = 5,
    ):
        self._config = config
        set_seed(self._config.seed)
        # filesystem paths
        self._output_dir = self._config.output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        self._results_file = os.path.join(self._output_dir, "results.json")
        self._metric_dir = os.path.join(self._output_dir, "metric")
        self._checkpoint_file = os.path.join(self._output_dir, "checkpoint.json")
        self._predictions_file = os.path.join(self._output_dir, "predictions.ndjson")

        # core components
        self.data: BaseHTMLDataset = config.dataset_config.create_dataset()
        self.pipeline: BasePipeline = config.pipeline_config.create_pipeline()
        self.evaluator: Evaluator = Evaluator(
            evaluation_metrics=self._config.dataset_config.evaluation_metrics,
            #TODO: needs to be parameters
            sample_eval_offload_every=1,
            sample_eval_resume=resume,
            sample_eval_offload_dir=self._metric_dir
        )
        self.tracker: BaseTracker = get_tracker(
            backend=tracker_backend,
            project=project_name,
            experiment_name=self._config.experiment_name,
            config=self._config.to_dict(),
        )


        # streaming & checkpoint parameters
        self._checkpoint_every = max(1, checkpoint_every)
        self._flush_every = max(1, flush_every)

        # minimal in-memory progress (no predictions stored here)
        self._progress = {
            "experiment_config": self._config.to_dict(),
            "last_batch": -1,
            "predictions_file": self._predictions_file,
        }

        if resume:
            self._load_checkpoint()
        else:
            # fresh run: reset predictions + checkpoint
            if os.path.exists(self._predictions_file):
                os.remove(self._predictions_file)
            if os.path.exists(self._checkpoint_file):
                print(f"[Experiment] Ignoring existing checkpoint: {self._checkpoint_file}")

        # connect references
        self.pipeline.set_experiment(self)
        self.data.set_experiment(self)
        self.evaluator.set_experiment(self)

        # graceful shutdown
        self._init_signals()

    # -------------------- Utilities --------------------
    def _init_signals(self) -> None:
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)

    def _graceful_shutdown(self, signum, frame) -> None:
        print(f"\n[Experiment] Received signal {signum}, saving checkpoint...")
        self._save_checkpoint()
        try:
            self.tracker.finish()
        except Exception:
            pass
        exit(0)

    def _json_default(self, obj):
        if torch is not None and isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None and isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    def _append_predictions_to_file(self, preds: List[Dict[str, Any]], file_handle) -> None:
        for p in preds:
            file_handle.write(
                json.dumps(p, default=self._json_default, ensure_ascii=False) + "\n"
            )

    def _predictions_iterator(self) -> Iterable["SamplePrediction"]:
        """
        Yield SamplePrediction instances from the NDJSON predictions file, line by line.

        Handles:
        - lines that are JSON objects (one prediction per line)
        - lines that are JSON arrays (multiple predictions in one line)
        - minor JSON decoding failures via ast.literal_eval fallback
        - missing keys by providing sensible defaults
        - coercing 'id' to str to avoid indexing/lookup mismatches
        """

        if not os.path.exists(self._predictions_file):
            # No predictions file yet -> empty iterator
            return
            yield  # make this a generator function even when returning early

        with open(self._predictions_file, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue

                # parse JSON, with fallback to ast.literal_eval for robustness
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        obj = ast.literal_eval(raw)
                    except Exception:
                        print(f"[Experiment] Warning: failed to parse predictions line {lineno}; skipping.")
                        continue

                # If an array was written on this line, iterate its items
                if isinstance(obj, list):
                    for item in obj:
                        if not isinstance(item, dict):
                            # wrap non-dict items
                            item = {"prediction": item}
                        # coerce id to string if present
                        if "id" in item and item["id"] is not None:
                            try:
                                item["id"] = str(item["id"])
                            except Exception:
                                pass
                        # ensure keys exist with defaults to avoid KeyError downstream
                        item.setdefault("query", None)
                        item.setdefault("ground_truth", None)
                        item.setdefault("prediction", None)
                        item.setdefault("filtered_html", None)
                        yield SamplePrediction(**item)
                    continue

                # If obj is a dict -> single prediction
                if not isinstance(obj, dict):
                    # wrap primitives into a prediction dict
                    obj = {"prediction": obj}

                if "id" in obj and obj["id"] is not None:
                    try:
                        obj["id"] = str(obj["id"])
                    except Exception:
                        pass

                obj.setdefault("query", None)
                obj.setdefault("ground_truth", None)
                obj.setdefault("prediction", None)
                obj.setdefault("filtered_html", None)

                yield SamplePrediction(**obj)


    # -------------------- Checkpointing --------------------
    def _save_checkpoint(self) -> None:
        atomic_write(self._checkpoint_file, {
            "experiment_config": self._progress["experiment_config"],
            "last_batch": self._progress["last_batch"],
            "predictions_file": self._progress["predictions_file"],
        })

    def _load_checkpoint(self) -> None:
        if os.path.exists(self._checkpoint_file):
            with open(self._checkpoint_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self._progress.update(
                last_batch=loaded.get("last_batch", -1),
                predictions_file=loaded.get("predictions_file", self._predictions_file),
            )
            print(f"[Experiment] Resuming from batch {self._progress['last_batch'] + 1}")

    # -------------------- Saving helpers --------------------
    def _save_results(self, results) -> None:
        with open(self._results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Experiment] Saved results to {self._results_file}")

    def _save_sample_eval(self, sample_eval, metric_name) -> None:
        """
        Save sample-level evaluations for a given metric in NDJSON format.
        Each sample evaluation is written as one JSON object per line.
        """
        os.makedirs(os.path.join(self._metric_dir, metric_name), exist_ok=True)
        file_name = os.path.join(self._metric_dir, metric_name, "sample_eval.ndjson")

        with open(file_name, "w", encoding="utf-8") as f:
            for entry in sample_eval:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[Experiment] Saved sample evaluation to {file_name}")


    def _save_config(self) -> None:
        config_file = os.path.join(self._output_dir, "experiment_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self._config.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Experiment] Saved experiment config to {config_file}")

    # -------------------- Main loop --------------------
    def run(self) -> Dict[str, Any]:
        set_seed(self._config.seed)
        batch_size = self._config.dataset_config.batch_size
        shuffle = self._config.dataset_config.shuffle

        if batch_size and hasattr(self.data, "batch_iterator"):
            iterator = self.data.batch_iterator(batch_size=batch_size, shuffle=shuffle)
            total = ceil(len(self.data) / batch_size)
        else:
            iterator = iter(self.data)
            total = len(self.data)

        results = None
        with open(self._predictions_file, "a", encoding="utf-8", buffering=1) as pred_f:
            try:
                for batch_idx, batch in enumerate(
                    tqdm(iterator, desc="Running Experiment", unit="batch", total=total)
                ):
                    if batch_idx <= self._progress["last_batch"]:
                        continue  # skip processed

                    try:
                        pred = self.pipeline.extract(batch)
                        # print("THIS IS THE PRED: ",pred)
                        if not isinstance(pred, list):
                            pred = list(pred)

                        self._append_predictions_to_file(pred, pred_f)

                        if hasattr(self.evaluator, "update"):
                            try:
                                #TODO: first evaluator call
                                results = self.evaluator.update(pred)
                            except Exception as e:
                                print(f"[Experiment] Evaluator update() failed: {e}")

                        self._progress["last_batch"] = batch_idx

                        if (batch_idx % self._flush_every) == 0:
                            pred_f.flush()
                            try:
                                os.fsync(pred_f.fileno())
                            except Exception:
                                pass

                        if (batch_idx % self._checkpoint_every) == 0:
                            self._save_checkpoint()

                    except Exception as e:
                        print(f"[Experiment] Error on batch {batch_idx}: {e}")
                        continue

                    finally:
                        del batch, pred
                        gc.collect()
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                print("[Experiment] Finished. Evaluating...")

                self._save_checkpoint()

                results = self.evaluator.get_final_result()


                self._save_results(results)
                # for metric in self.evaluator.get_metrics():
                #     self._save_sample_eval(metric._sample_eval, metric.name())
                self._save_config()

                try:
                    self.tracker.log_metrics(results, step=self._progress["last_batch"] + 1)
                except Exception:
                    pass

                return {"predictions_file": self._predictions_file, "results": results}

            finally:
                try:
                    self.tracker.finish()
                except Exception:
                    pass
