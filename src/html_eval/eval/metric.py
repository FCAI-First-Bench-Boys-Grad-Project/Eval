from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING, Iterable, Optional
from abc import ABC, abstractmethod
import os
import json

from html_eval.util.eval_util import is_not_null, repair_and_parse, compute_f1_squad
from html_eval.eval.matcher import Matcher, MatcherConfig
from html_eval.core.types import SamplePrediction, SampleEvaluation

if TYPE_CHECKING:
    from html_eval.eval.evaluator import Evaluator

METRICS_REGISTRY: dict[str, type["Metric"]] = {}


def register_metric(name: str):
    def wrapper(cls):
        METRICS_REGISTRY[name] = cls
        return cls
    return wrapper


class Metric(ABC):
    def __init__(self, evaluator: "Evaluator"):
        super().__init__()

        # aggregated values for this metric (populated by update/calculate)
        self._values: Dict[str, Any] = {}
        # small in-memory buffer of sample evaluations (kept bounded)
        self._sample_eval: List[SampleEvaluation] = []
        self._sample_eval_buffer: List[SampleEvaluation] = []

        # offload configuration
        self._offload_active: bool = False
        self._offload_dir: Optional[str] = None
        self._offload_path: Optional[str] = None
        self._offload_every: int = 100  # default buffer size before writing
        self._offload_resume: bool = False

        # reference back to evaluator
        self.evaluator = evaluator

    @property
    def values(self) -> Dict[str, Any]:
        """
        Returns the aggregated values of the metric.
        """
        return self._values

    @property
    def sample_evaluations(self) -> List[SampleEvaluation]:
        """
        Returns the in-memory sample evaluations buffer (bounded).
        Use iter_sample_evals() to iterate over all persisted + buffered entries.
        """
        return self._sample_eval

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def calculate(self, predictions: List[SamplePrediction]) -> Dict[str, Any]:
        """
        One-shot calculation produced by the metric. Typically implemented by
        resetting internal state and calling update on the entire dataset.
        """
        ...

    # ------------ offload helpers ------------
    def configure_sample_offload(self, offload_dir: Optional[str], *, offload_every: int = 100, resume: bool = False):
        """
        Enable on-disk offload of sample evaluations. If offload_dir is None, no offload is used.

        - offload_every: how many sample-evals to buffer before writing to disk.
        - resume: if True, we will append to an existing offload file if present.
                  if False, existing offload file is truncated.
        """
        if offload_dir is None:
            self._offload_active = False
            self._offload_dir = None
            self._offload_path = None
            return

        os.makedirs(offload_dir, exist_ok=True)
        self._offload_active = True
        self._offload_dir = offload_dir
        self._offload_every = max(1, int(offload_every))
        self._offload_resume = bool(resume)
        file_name = f"{self.name()}_sample_eval.ndjson"
        self._offload_path = os.path.join(offload_dir, file_name)

        # if not resuming, truncate existing file so we start fresh
        if not self._offload_resume and os.path.exists(self._offload_path):
            try:
                # atomic overwrite
                with open(self._offload_path, "w", encoding="utf-8") as f:
                    pass
            except Exception:
                pass

    def _serialize_sample_eval(self, sample_eval: SampleEvaluation) -> Dict[str, Any]:
        """
        Convert a SampleEvaluation object to a JSON-serializable dict.
        We try dataclasses.asdict, then __dict__, then fallback to manual extraction.
        """
        try:
            # avoid importing dataclasses at module import time if not needed
            from dataclasses import asdict
            try:
                return asdict(sample_eval)
            except Exception:
                # asdict may fail if not a dataclass
                pass
        except Exception:
            pass

        # fallback to __dict__
        if hasattr(sample_eval, "__dict__"):
            d = sample_eval.__dict__
            # ensure that nested non-serializable objects are converted to strings
            cleaned = {}
            for k, v in d.items():
                try:
                    json.dumps(v)
                    cleaned[k] = v
                except Exception:
                    try:
                        cleaned[k] = str(v)
                    except Exception:
                        cleaned[k] = None
            return cleaned

        # last resort: return a best-effort dict
        return {
            "id": getattr(sample_eval, "id", None),
            "query": getattr(sample_eval, "query", None),
            "ground_truth": getattr(sample_eval, "ground_truth", None),
            "prediction": getattr(sample_eval, "prediction", None),
            "filtered_html": getattr(sample_eval, "filtered_html", None),
            "evaluation": getattr(sample_eval, "evaluation", None),
        }

    def append_sample_eval(self, sample_eval: SampleEvaluation):
        """
        Append a sample-eval entry. If offload is active we buffer and flush to disk when necessary.
        If offload is not active we keep it in _sample_eval (legacy behavior).
        """
        if self._offload_active:
            self._sample_eval_buffer.append(sample_eval)
            # keep a tiny in-memory sample list for quick access (bounded)
            # store only last few samples (not all) to avoid memory growth:
            self._sample_eval.append(sample_eval)
            if len(self._sample_eval) > max(100, self._offload_every // 10):
                # drop oldest in-memory sample evals to bound memory usage
                del self._sample_eval[: len(self._sample_eval) - max(100, self._offload_every // 10)]
            if len(self._sample_eval_buffer) >= self._offload_every:
                self._flush_buffer_to_disk()
        else:
            # legacy: keep everything in memory
            self._sample_eval.append(sample_eval)

    def _flush_buffer_to_disk(self) -> Optional[str]:
        """
        Internal: write buffered sample-evals to the offload NDJSON file.
        Returns the offload path or None if not active.
        """
        if not self._offload_active:
            return None
        if not self._sample_eval_buffer:
            return self._offload_path

        path = self._offload_path
        try:
            with open(path, "a", encoding="utf-8") as f:
                for entry in self._sample_eval_buffer:
                    obj = self._serialize_sample_eval(entry)
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            # clear buffer after successful write
            self._sample_eval_buffer = []
            return path
        except Exception:
            # on failure keep buffer in memory for next attempt
            return path

    def flush_sample_evals_to_disk(self, force: bool = False):
        """
        Write buffered sample evaluations to disk in NDJSON format.
        If offloading is enabled, appends each sample evaluation as one line.
        """
        if not self._offload_active or not self._offload_path:
            return

        if not self._sample_eval_buffer:
            return

        if force or len(self._sample_eval_buffer) >= self._offload_every:
            try:
                with open(self._offload_path, "a", encoding="utf-8") as f:
                    for entry in self._sample_eval_buffer:
                        f.write(json.dumps(self._serialize_sample_eval(entry), ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Metric] Warning: failed to offload sample evals for {self.name()}: {e}")
            finally:
                self._sample_eval_buffer.clear()


    def sample_eval_path(self) -> Optional[str]:
        """
        Get the NDJSON file path used for offloading sample evaluations for this metric.
        """
        return self._offload_path

    def iter_sample_evals(self) -> Iterable[Dict[str, Any]]:
        """
        Iterate over persisted sample-evals (disk) followed by any buffered in-memory items.
        Yields dicts (JSON-serializable).
        """
        # first yield persisted entries if file available
        if self._offload_active and self._offload_path and os.path.exists(self._offload_path):
            try:
                with open(self._offload_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception:
                            # fallback: yield raw string if parsing failed
                            yield {"raw": line}
            except Exception:
                # ignore read errors and yield buffered items below
                pass

        # then yield any buffered in-memory sample evaluations (converted to dicts)
        for entry in self._sample_eval_buffer:
            yield self._serialize_sample_eval(entry)

        # lastly yield the in-memory _sample_eval (bounded), but avoid duplication of what we persisted
        for entry in self._sample_eval:
            yield self._serialize_sample_eval(entry)

    # ------------ incremental API ------------
    def reset(self) -> None:
        """
        Reset internal accumulators. Subclasses should call super().reset() if overriding.
        IMPORTANT: when offload is active we don't delete the persisted file here (preserve data)
        unless the caller explicitly wants to truncate.
        """
        self._values = {}
        # clear in-memory buffers (persisted file remains unless user truncates)
        self._sample_eval = []
        self._sample_eval_buffer = []

    def update(self, predictions: List[SamplePrediction]) -> Dict[str, Any]:
        """
        Default incremental update: by default call calculate (stateless).
        Subclasses should override for efficient incremental updates.
        Should return an aggregated dict similar to calculate(...) describing current aggregated metrics.
        """
        # By default, recalc everything from scratch (safe fallback)
        self.reset()
        return self.calculate(predictions)

    def stream(self, predictions: Iterable[SamplePrediction], batch_size: int = 1):
        """
        Default streaming implementation: take an iterable, call update per batch,
        and yield aggregated dicts after each batch.
        """
        batch: List[SamplePrediction] = []
        for item in predictions:
            batch.append(item)
            if len(batch) >= batch_size:
                yield self.update(batch)
                # after each batch ensure sample-evals are persisted to disk if active
                self.flush_sample_evals_to_disk(force=True)
                batch = []
        if batch:
            yield self.update(batch)
            self.flush_sample_evals_to_disk(force=True)

    def summary(self) -> Dict[str, Any]:
        """
        Returns a consistent summary dict representing the current aggregated values.
        Subclasses should ensure _values is populated.
        """
        return self._values
        

@register_metric("token_f1")
class TokenF1(Metric):
    def __init__(self, evaluator):
        super().__init__(evaluator)
        self._count: int = 0
        self._sum_f1: float = 0.0
        self._sum_precision: float = 0.0
        self._sum_recall: float = 0.0

    def name(self) -> str:
        return "token_f1"

    def _normalize_into_string(self, value: Any) -> str:
        """
        Normalize any input into a string to avoid dtype issues and ensure stable F1 computation.
        - None -> ""
        - dict/list -> JSON string
        - everything else -> str()
        """
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    def reset(self) -> None:
        super().reset()
        self._count: int = 0
        self._sum_f1: float = 0.0
        self._sum_precision: float = 0.0
        self._sum_recall: float = 0.0

    def calculate(self, predictions: List[SamplePrediction]) -> Dict[str, Any]:
        """
        One-shot calculation: reset state, update with provided predictions, return aggregated dict.
        """
        self.reset()
        self.update(predictions)
        return {
            "f1": {"average": self._values.get("f1", 0.0)},
            "precision": {"average": self._values.get("precision", 0.0)},
            "recall": {"average": self._values.get("recall", 0.0)},
        }

    def update(self, predictions: List[SamplePrediction]) -> Dict[str, Any]:
        """
        Incrementally update running sums and sample_evals from a batch of predictions.
        """
        if not predictions:
            return {
                "f1": {"average": self._values.get("f1", 0.0)},
                "precision": {"average": self._values.get("precision", 0.0)},
                "recall": {"average": self._values.get("recall", 0.0)},
            }

        for pred in predictions:
            pred_str = self._normalize_into_string(pred.prediction)
            gold_str = self._normalize_into_string(pred.ground_truth)

            try:
                f1, prec, rec = compute_f1_squad(gold_str, pred_str)
            except Exception:
                f1, prec, rec = 0.0, 0.0, 0.0

            self._count += 1
            self._sum_f1 += float(f1)
            self._sum_precision += float(prec)
            self._sum_recall += float(rec)

            eval_entry = SampleEvaluation(
                id=pred.id,
                query=pred.query,
                ground_truth=pred.ground_truth,
                prediction=pred.prediction,
                filtered_html=getattr(pred, "filtered_html", None),
                evaluation={"f1": float(f1), "precision": float(prec), "recall": float(rec)}
            )
            # append via offload-aware helper
            self.append_sample_eval(eval_entry)

        # update aggregated values
        if self._count > 0:
            self._values["f1"] = self._sum_f1 / self._count
            self._values["precision"] = self._sum_precision / self._count
            self._values["recall"] = self._sum_recall / self._count
        else:
            self._values["f1"] = 0.0
            self._values["precision"] = 0.0
            self._values["recall"] = 0.0

        return {
            "f1": {"average": self._values["f1"]},
            "precision": {"average": self._values["precision"]},
            "recall": {"average": self._values["recall"]},
        }

    def stream(self, predictions: Iterable[SamplePrediction], batch_size: int = 1):
        """
        Streaming: yield aggregated dict after each batch.
        """
        batch: List[SamplePrediction] = []
        for item in predictions:
            batch.append(item)
            if len(batch) >= batch_size:
                yield self.update(batch)
                # ensure writes persisted
                self.flush_sample_evals_to_disk(force=True)
                batch = []
        if batch:
            yield self.update(batch)
            self.flush_sample_evals_to_disk(force=True)


@register_metric("page_level_f1")
class PageLevelF1(Metric):
    def name(self) -> str:
        return "page_level_f1"

    def __init__(self, evaluator: "Evaluator"):
        super().__init__(evaluator)
        # per-website per-field counters
        # structure: { website: { field: {page_hits, extracted_pages, ground_truth_pages} } }
        self._website_field_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._fields: set = set()
        self._matcher: Matcher = Matcher(MatcherConfig(is_fuzzy=False))

    def reset(self) -> None:
        super().reset()
        self._website_field_counts = {}
        self._fields = set()
        self._values = {}

    def _ensure_field_entry(self, website: str, field: str):
        if website not in self._website_field_counts:
            self._website_field_counts[website] = {}
        if field not in self._website_field_counts[website]:
            self._website_field_counts[website][field] = {
                'page_hits': 0,
                'extracted_pages': 0,
                'ground_truth_pages': 0
            }

    def _update_counts_for_sample(self, website: str, field: str, pred, gt, pred_match_gt: bool):
        self._ensure_field_entry(website, field)
        if pred_match_gt:
            self._website_field_counts[website][field]['page_hits'] += 1

        if is_not_null(pred) and ((field not in pred) or is_not_null(pred.get(field))):
            self._website_field_counts[website][field]['extracted_pages'] += 1

        if is_not_null(gt) and ((field not in gt) or is_not_null(gt.get(field))):
            self._website_field_counts[website][field]['ground_truth_pages'] += 1

    def _compute_aggregations(self):
        results = self._website_field_counts

        # Per-website aggregation
        website_aggregated_results: Dict[str, Dict[str, float]] = {}
        for website, fields in results.items():
            f1_website = 0.0
            precision_website = 0.0
            recall_website = 0.0
            total_fields = len(fields)
            for field, metrics in fields.items():
                extracted = metrics['extracted_pages']
                ground_truth = metrics['ground_truth_pages']
                page_hits = metrics['page_hits']

                if extracted == 0 or ground_truth == 0:
                    f1 = 0.0
                    precision = 0.0
                    recall = 0.0
                else:
                    precision = page_hits / extracted if extracted > 0 else 0.0
                    recall = page_hits / ground_truth if ground_truth > 0 else 0.0
                    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                f1_website += f1
                precision_website += precision
                recall_website += recall

            website_aggregated_results[website] = {
                'f1': (f1_website / total_fields) if total_fields > 0 else 0.0,
                'precision': (precision_website / total_fields) if total_fields > 0 else 0.0,
                'recall': (recall_website / total_fields) if total_fields > 0 else 0.0
            }

        # Per-field aggregation across websites (normal mean)
        field_aggregated_results: Dict[str, Dict[str, float]] = {}
        for field in self._fields:
            f1_field = 0.0
            precision_field = 0.0
            recall_field = 0.0
            total_websites = 0
            for website in results:
                if field in results[website]:
                    # ensure computing field-level f1 if present
                    wf = results[website][field]
                    # If earlier code already computed 'f1' keys, use them; otherwise compute per website/field
                    if 'f1' in wf:
                        f1_field += wf.get('f1', 0.0)
                        precision_field += wf.get('precision', 0.0)
                        recall_field += wf.get('recall', 0.0)
                    else:
                        # compute from counts
                        extracted = wf.get('extracted_pages', 0)
                        ground_truth = wf.get('ground_truth_pages', 0)
                        page_hits = wf.get('page_hits', 0)
                        if extracted == 0 or ground_truth == 0:
                            pf, pr, fr = 0.0, 0.0, 0.0
                        else:
                            pf = page_hits / extracted if extracted > 0 else 0.0
                            pr = page_hits / ground_truth if ground_truth > 0 else 0.0
                            fr = (2 * pf * pr) / (pf + pr) if (pf + pr) > 0 else 0.0
                        f1_field += fr
                        precision_field += pf
                        recall_field += pr
                    total_websites += 1
            field_aggregated_results[field] = {
                'f1': (f1_field / total_websites) if total_websites > 0 else 0.0,
                'precision': (precision_field / total_websites) if total_websites > 0 else 0.0,
                'recall': (recall_field / total_websites) if total_websites > 0 else 0.0
            }

        # Overall aggregation (mean across websites)
        f1_overall = 0.0
        precision_overall = 0.0
        recall_overall = 0.0
        total_websites = len(website_aggregated_results)
        for _, metrics in website_aggregated_results.items():
            f1_overall += metrics['f1']
            precision_overall += metrics['precision']
            recall_overall += metrics['recall']

        self._values['f1'] = (f1_overall / total_websites) if total_websites > 0 else 0.0
        self._values['precision'] = (precision_overall / total_websites) if total_websites > 0 else 0.0
        self._values['recall'] = (recall_overall / total_websites) if total_websites > 0 else 0.0
        self._values['website_aggregated_results'] = website_aggregated_results
        self._values['field_aggregated_results'] = field_aggregated_results
        self._values['results'] = results

    def calculate(self, predictions: List[SamplePrediction]) -> Dict[str, Any]:
        """
        One-shot calculation: reset internal counts and process the provided predictions
        as if they were the entire dataset.
        """
        self.reset()
        self.update(predictions)
        # make sure everything persisted
        self.flush_sample_evals_to_disk(force=True)
        return {
            "f1": {"average": self._values.get('f1', 0.0)},
            "precision": {"average": self._values.get('precision', 0.0)},
            "recall": {"average": self._values.get('recall', 0.0)},
        }

    def update(self, predictions: List[SamplePrediction]) -> Dict[str, Any]:
        """
        Process a batch of SamplePrediction and update internal counters.
        Returns the aggregated dict (same structure as calculate).
        """
        if not predictions:
            return {
                "f1": {"average": self._values.get('f1', 0.0)},
                "precision": {"average": self._values.get('precision', 0.0)},
                "recall": {"average": self._values.get('recall', 0.0)},
            }

        for pred in predictions:
            try:
                gt = repair_and_parse(pred.ground_truth)
            except Exception:
                gt = {}

            # ensure prediction is accessible as mapping/dict
            pred_val = pred.prediction
            if isinstance(pred_val, str):
                try:
                    pred_val = repair_and_parse(pred_val)
                except Exception:
                    pred_val = {}

            # schema fields are those appearing in ground truth (fallback empty)
            fields = list(gt.keys()) if isinstance(gt, dict) else []

            # website lookup: prefer experiment hook (if available)
            website = str(getattr(pred, "id", "unknown")).split('_')[0]

            # process each field
            for field in fields:
                self._fields.add(field)
                try:
                    pred_match_gt = self._matcher.compare(gt.get(field), (pred_val or {}).get(field))
                except Exception:
                    pred_match_gt = False

                self._update_counts_for_sample(website, field, pred_val, gt, pred_match_gt)

            # create sample evaluation mapping
            current_sample_eval = {}
            for field in fields:
                try:
                    pred_match_gt = self._matcher.compare(gt.get(field), (pred_val or {}).get(field))
                except Exception:
                    pred_match_gt = False
                current_sample_eval[field] = 1 if pred_match_gt else 0

            eval_entry = SampleEvaluation(
                id=pred.id,
                query=pred.query,
                ground_truth=pred.ground_truth,
                prediction=pred.prediction,
                filtered_html=getattr(pred, "filtered_html", None),
                evaluation=current_sample_eval
            )

            # offload-aware append
            self.append_sample_eval(eval_entry)

        # recompute aggregated metrics after this batch
        self._compute_aggregations()

        return {
            "f1": {"average": self._values.get('f1', 0.0)},
            "precision": {"average": self._values.get('precision', 0.0)},
            "recall": {"average": self._values.get('recall', 0.0)},
        }

    def stream(self, predictions: Iterable[SamplePrediction], batch_size: int = 1):
        """
        Streaming processing - yields aggregated metrics after each processed batch.
        """
        batch: List[SamplePrediction] = []
        for item in predictions:
            batch.append(item)
            if len(batch) >= batch_size:
                res = self.update(batch)
                # persist sample-evals of this batch
                self.flush_sample_evals_to_disk(force=True)
                yield res
                batch = []
        if batch:
            res = self.update(batch)
            self.flush_sample_evals_to_disk(force=True)
            yield res
