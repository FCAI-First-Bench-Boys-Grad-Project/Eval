from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable
import os

from html_eval.core.types import SamplePrediction
from html_eval.eval.metric import Metric, METRICS_REGISTRY
from html_eval.eval.matcher import MatcherConfig


class Evaluator:

    def __init__(
        self,
        evaluation_metrics: List[str],
        matcher_cfg: Optional[MatcherConfig] = None,
        *,
        # optional disk offload for per-metric sample evaluations
        sample_eval_offload_dir: Optional[str] = None,
        sample_eval_offload_every: int = 100,
        sample_eval_resume: bool = False,
    ):

        self.evaluation_metrics = evaluation_metrics
        self.matcher_cfg = matcher_cfg or MatcherConfig()

        # instantiate metric objects
        self.metrics: List[Metric] = []
        for name in evaluation_metrics:
            if name not in METRICS_REGISTRY:
                raise KeyError(f"Unknown metric {name}")
            self.metrics.append(METRICS_REGISTRY[name](self))

        # sample-eval offload configuration (optional)
        self.sample_eval_offload_dir = sample_eval_offload_dir
        self.sample_eval_offload_every = int(sample_eval_offload_every)
        self.sample_eval_resume = bool(sample_eval_resume)

        if self.sample_eval_offload_dir:
            os.makedirs(self.sample_eval_offload_dir, exist_ok=True)
            for metric in self.metrics:
                metric.configure_sample_offload(
                    offload_dir=self.sample_eval_offload_dir,
                    offload_every=self.sample_eval_offload_every,
                    resume=self.sample_eval_resume,
                )

        # will be assigned by experiment via set_experiment
        self.experiment = None

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

    # ---------------- one-shot evaluate ----------------
    def evaluate(self, predictions: List[SamplePrediction]) -> Dict[str, Any]:
        """
        One-shot evaluation over a list of SamplePrediction.
        Delegates to metric.calculate(...) (keeps backward compatibility).
        """
        results: Dict[str, Any] = {}
        for metric in self.metrics:
            res = metric.calculate(predictions)
            # after calculate, make sure any buffered sample_evals are persisted
            metric.flush_sample_evals_to_disk(force=True)
            results[metric.name()] = res
        return results

    # ---------------- incremental update ----------------
    def update(self, predictions: List[SamplePrediction]) -> Dict[str, Any]:
        """
        Incrementally update metrics with a batch of predictions.
        Each metric is expected to implement .update(...) which updates its internal
        running state and returns the current aggregated result for that metric.

        After updating, we flush the metric's sample-eval buffer to disk to avoid
        memory growth across many batches.
        """
        results: Dict[str, Any] = {}
        for metric in self.metrics:
            res = metric.update(predictions)
            # ensure we flush per-batch so memory remains bounded
            metric.flush_sample_evals_to_disk(force=True)
            results[metric.name()] = res
        return results

    # ---------------- streaming interfaces ----------------
    def stream(self, predictions: Iterable[SamplePrediction], batch_size: int = 256):
        """
        Generator that yields intermediate aggregated metric results after each
        processed batch. Useful for progressive logging.
        Yields tuples: (batch_index, {metric_name: aggregated_result, ...})
        """
        batch = []
        batch_idx = 0
        for item in predictions:
            batch.append(item)
            if len(batch) >= batch_size:
                batch_idx += 1
                batch_res = self.update(batch)
                yield batch_idx, batch_res
                batch = []
        # final partial batch
        if batch:
            batch_idx += 1
            batch_res = self.update(batch)
            yield batch_idx, batch_res

    def get_final_result(self) -> Dict[str,Any]:
        # after consuming entire stream, collect final aggregated values
        final_results: Dict[str, Any] = {}
        for metric in self.metrics:
            final_results[metric.name()] = metric.summary()
        return final_results

    def evaluate_stream(self, predictions: Iterable[SamplePrediction], batch_size: int = 256) -> Dict[str, Any]:
        """
        Consume an iterator of SamplePrediction (e.g. from disk) in batches and
        return final aggregated metrics. Uses .update() internally.
        """
        for _idx, batch_res in self.stream(predictions, batch_size=batch_size):
            # we simply consume generator; update already accumulates state in metrics
            pass

        return self.get_final_result()

    # ---------------- helpers to locate sample-eval files ----------------
    def get_sample_eval_files(self) -> Dict[str, str]:
        """
        Return a mapping metric_name -> path to NDJSON file (if the metric offloads).
        Metrics that do not offload will not appear in the mapping.
        """
        files: Dict[str, str] = {}
        for metric in self.metrics:
            p = metric.sample_eval_path()
            if p:
                files[metric.name()] = p
        return files

    def iter_all_sample_evals(self):
        """
        Iterate over all sample evaluations for all metrics.
        Yields tuples (metric_name, sample_eval_dict)
        """
        for metric in self.metrics:
            for entry in metric.iter_sample_evals():
                yield metric.name(), entry
