# master_evaluator.py
from __future__ import annotations
import re
import string
import ast
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import Counter, OrderedDict
import pandas as pd
import polars as pl
from abc import ABC, abstractmethod
import concurrent.futures
import math
import os

# Try to use rapidfuzz (faster) else fallback to fuzzywuzzy
try:
    from rapidfuzz import fuzz as _rfuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False
    try:
        from fuzzywuzzy import fuzz as _fwuzz  # type: ignore
    except Exception:
        _fwuzz = None

# --------------------------
# Configs / Utilities
# --------------------------
@dataclass
class MatcherConfig:
    is_fuzzy: bool = False
    fuzzy_threshold: int = 90  # 0-100

def is_not_null(x: Any) -> bool:
    """Robustly detect non-null value (supports pandas/polars/list/scalars)."""
    if x is None:
        return False
    # string
    if isinstance(x, str):
        if x == "<NULL>" or x.strip() == "":
            return False
    # pandas
    if isinstance(x, pd.Series):
        if x.empty:
            return False
        return not x.isna().all()
    # polars
    try:
        if isinstance(x, pl.Series):
            return not x.to_pandas().isna().all()
    except Exception:
        pass
    # list/tuple
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return False
        try:
            return not pd.Series(x, dtype="object").isna().all()
        except Exception:
            return any(item is not None for item in x)
    # scalar
    try:
        return not pd.isna(x)
    except Exception:
        return True

# --------------------------
# Text normalization & SQuAD tokenization (for token-level F1)
# --------------------------
def _squad_normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lower, remove punctuation, articles, extra whitespace."""
    if s is None:
        return ""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _squad_get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return _squad_normalize_answer(s).split()

def compute_f1_squad(a_gold: str, a_pred: str) -> Tuple[float, float, float]:
    """returns f1, precision, recall for SQuAD-style answers"""
    gold_toks = _squad_get_tokens("" if a_gold is None else str(a_gold))
    pred_toks = _squad_get_tokens("" if a_pred is None else str(a_pred))
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        if gold_toks == pred_toks:
            return 1.0, 1.0, 1.0  # <-- always return a tuple
        else:
            return 0.0, 0.0, 0.0  # <-- always return a tuple
    
    if num_same == 0:
        return 0.0, 0.0, 0.0

    prec = num_same / len(pred_toks)
    rec = num_same / len(gold_toks)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec

    

# --------------------------
# Matcher (exact / fuzzy) # TODO: Add Embedder Evaluation
# --------------------------
class Matcher:
    def __init__(self, cfg: Optional[MatcherConfig] = None):
        self.cfg = cfg or MatcherConfig()

    def _normalize_gt(self, gt: Any) -> List[Any]:
        """
        Turn GT into list of candidates:
         - If list -> return list filtered for non-null
         - If string begins with '[' parse literal -> list
         - Else single-item list
        """
        if not is_not_null(gt) and not (isinstance(gt, str) and gt.strip().startswith("[")):
            return []
        if isinstance(gt, list):
            items = gt
        elif isinstance(gt, str):
            try:
                evaluated = ast.literal_eval(gt)
                if isinstance(evaluated, list):
                    items = evaluated
                else:
                    items = [evaluated]
            except Exception:
                items = [gt]
        else:
            items = [gt]
        return [it for it in items if is_not_null(it)]

    def compare(self, gt: Any, pred: Any) -> bool:
        """
        Return True if pred matches any gt candidate.
        Exact match by default, fuzzy if configured.
        """
        if not is_not_null(pred):
            return False
        candidates = self._normalize_gt(gt)
        if not candidates:
            return False
        pred_s = str(pred)
        if self.cfg.is_fuzzy:
            threshold = max(0, min(100, self.cfg.fuzzy_threshold))
            for c in candidates:
                try:
                    cs = str(c)
                    if _HAS_RAPIDFUZZ:
                        score = _rfuzz.ratio(cs.lower(), pred_s.lower())
                    else:
                        if _fwuzz is None:
                            # no fuzzy lib installed -> fallback to simple equality
                            continue
                        score = _fwuzz.ratio(cs.lower(), pred_s.lower())
                    if score >= threshold:
                        return True
                except Exception:
                    continue
            return False
        else:
            for c in candidates:
                if isinstance(pred, (int, float)) and isinstance(c, (int, float)):
                    if pred == c:
                        return True
                if str(c) == pred_s:
                    return True
            return False


# --------------------------
# Metrics base class
# --------------------------
class Metric(ABC):
    def __init__(self,evaluator: Evaluator):
        super().__init__()

        
        self._values: Dict[str, Any] = {}
        self._scores: Dict[str, List[float]] = {}
        self.evaluator = evaluator  # Reference to the Evaluator instance

    @property
    def values(self) -> Dict[str, Any]:
        """
        Returns the aggregated values of the metric.
        """
        return self._values
    
    @property
    def scores(self) -> Dict[str, List[float]]:
        """
        Returns the individual scores for each instance.
        """
        return self._scores

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def calculate(self,
                  pred: Any,
                  gt: Any,
                  parallel: bool = False,
                  n_procs: int = 1) -> Dict[str, Any]:
       
        ...


class TokenF1(Metric):
    def name(self) -> str:
        return "token_f1"
    
    def calculate(self,
                  pred: pl.Series,
                  gt: pl.Series,
                  parallel: bool = False,
                  n_procs: int = 1) -> Dict[str, Any]:
        """
        Calculate token-level F1 score for SQuAD-style answers.
        """
        # F1 calculation for each pair
        def _calculate_f1(row: tuple[str, str]) -> Dict[str, float]:
            a_pred, a_gold = row
            f1, prec, rec = compute_f1_squad(a_gold, a_pred)
            return {"f1": f1, "precision": prec, "recall": rec}

        rows = zip(pred.to_list(), gt.to_list())

        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor:
                results = list(executor.map(_calculate_f1, rows))
        else:
            results = [_calculate_f1(row) for row in rows]

        # Aggregate
        f1_scores = [res["f1"] for res in results]
        precision_scores = [res["precision"] for res in results]
        recall_scores = [res["recall"] for res in results]  

        # Save into self._values and self._scores
        self._values["f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        self._values["precision"] = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        self._values["recall"] = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

        self._scores["f1"] = f1_scores
        self._scores["precision"] = precision_scores
        self._scores["recall"] = recall_scores

        return {
            "f1": {
            'average': self._values["f1"],
            },
            "precision": {
            'average': self._values["precision"],
            },
            "recall": {
            'average': self._values["recall"],
            }
        }


class PageLevelF1(Metric):
    def name(self) -> str:
        return "page_level_f1"
    
    def calculate(self,
                  pred: pl.Series,
                  gt: pl.Series,
                  parallel: bool = False,
                  n_procs: int = 1) -> Dict[str, Any]:
        """
        Calculate page-level F1 score for json answers.
        Implementing both Markuplm and swde evaluation method.
        
        I assume that pred and gt are polars Series containing Dict[str, Any] or similar structures.       
        """

        # Get the schema of the Domain
        schema = gt.keys()
        '''
        results should contain:
        website -> field -> {
            'page_hits': int,
            'extracted_pages': int,
            'ground_truth_pages': int,
            'f1': float,
            'precision': float,
            'recall': float
        }
        '''
        results = {}
        em_matcher = Matcher(MatcherConfig(is_fuzzy=False))
        # fuzzy_matcher = Matcher(MatcherConfig(is_fuzzy=True, fuzzy_threshold=90)) # Maybe for later use

        # TODO: might try to find a faster way to do this later
        # Collect results per website and per field
        for idx, (pred, gt) in enumerate(zip(pred.to_list(), gt.to_list())):
            if not is_not_null(pred) or not is_not_null(gt):
                continue
            
            current_website = self.evaluator.experiment.data[idx]['website_name']
            # Iterate over each field in the schema
            for field in schema:
                pred_match_gt = em_matcher.compare(gt[field], pred[field])
                if pred_match_gt:
                    results[current_website][field]['page_hits'] += 1
                if is_not_null(pred[field]):
                    results[current_website][field]['extracted_pages'] += 1
                if is_not_null(gt[field]):
                    results[current_website][field]['ground_truth_pages'] += 1

        # Calculate the scores for each field in each website
        for website in results:
            for field in results[website]:
                extracted = results[website][field]['extracted_pages']
                ground_truth = results[website][field]['ground_truth_pages']
                page_hits = results[website][field]['page_hits']

                if extracted == 0 or ground_truth == 0:
                    f1 = 0.0
                    precision = 0.0
                    recall = 0.0
                else:
                    precision = page_hits / extracted
                    recall = page_hits / ground_truth
                    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                results[website][field]['f1'] = f1
                results[website][field]['precision'] = precision
                results[website][field]['recall'] = recall
        '''
        What I want to do is get the average of metrics for each 
        '''
        # Aggregate results

        # Per-website aggregation
        website_aggregated_results = {}
        for website, fields in results.items():
            f1_website = 0.0
            precision_website = 0.0
            recall_website = 0.0
            total_fields = len(fields)
            for field, metrics in fields.items():
                f1_website += metrics['f1']
                precision_website += metrics['precision']
                recall_website += metrics['recall']
            website_aggregated_results[website] = {
                'f1': f1_website / total_fields if total_fields > 0 else 0.0,
                'precision': precision_website / total_fields if total_fields > 0 else 0.0,
                'recall': recall_website / total_fields if total_fields > 0 else 0.0
            }
        
        # Per-field aggregation
        # Goes through each field in the schema and averages the metric across all websites
        '''
        Camera --> (price , model , brand)
        -> amazon for each attribute (f1, precision, recall)
        -> bestbuy for each attribute (f1, precision, recall)
        -> walmart for each attribute (f1, precision, recall)

        now we need scores for attributes across websites
        Normal Mean 
        f1_price = (f1_price_amazon + f1_price_bestbuy + f1_price_walmart) / 3
        Weighted Mean TODO: Add it later
        f1_price = (f1_price_amazon * n_amazon + f1_price_bestbuy * n_bestbuy + f1_price_walmart * n_walmart) / (n_amazon + n_bestbuy + n_walmart)
        '''

        # This aggregates using Normal Mean 
        field_aggregated_results = {}
        for field in schema:
            f1_field = 0.0
            precision_field = 0.0
            recall_field = 0.0
            total_websites = 0
            for website in results:
                f1_field += results[website][field]['f1']
                precision_field += results[website][field]['precision']
                recall_field += results[website][field]['recall']
                total_websites += 1
            field_aggregated_results[field] = {
                'f1': f1_field / total_websites if total_websites > 0 else 0.0,
                'precision': precision_field / total_websites if total_websites > 0 else 0.0,
                'recall': recall_field / total_websites if total_websites > 0 else 0.0
            }

        # Overall aggregation
        f1_overall = 0.0
        precision_overall = 0.0
        recall_overall = 0.0
        total_websites = len(website_aggregated_results)
        for website, metrics in website_aggregated_results.items():
            f1_overall += metrics['f1']
            precision_overall += metrics['precision']
            recall_overall += metrics['recall']
        self._values['f1'] = f1_overall / total_websites if total_websites > 0 else 0.0
        self._values['precision'] = precision_overall / total_websites if total_websites > 0 else 0.0
        self._values['recall'] = recall_overall / total_websites if total_websites > 0 else 0.0 

        self._scores = {
            'website_aggregated_results': website_aggregated_results,
            'field_aggregated_results': field_aggregated_results,
            'overall': self._values
        }

        return {
            "f1": {
                'average': self._values['f1'],
            },
            "precision": {
                'average': self._values['precision'],
            },
            "recall": {
                'average': self._values['recall'],
            }
        }

                
            
                
        
                

            

                
        


# --------------------------
# Evaluator facade
# --------------------------
class Evaluator:
    METRIC_CLASSES = {
        "token_f1": TokenF1,
        # "em": ExatMatch,  # TODO: Implement ExactMatch
        "page_level_f1": PageLevelF1,
    }

    def __init__(self, evaluation_metrics: List[str],
                 matcher_cfg: Optional[MatcherConfig] = None,
                 parallel: bool = False,
                 n_procs: int = 1):
        """
        parallel: whether to use process-based parallelism
        n_procs: number of processes to use when parallel=True
        """
        self.evaluation_metrics = evaluation_metrics
        self.matcher_cfg = matcher_cfg or MatcherConfig()
        self.parallel = parallel
        self.n_procs = max(1, n_procs)
        # instantiate metric objects
        self.metrics = [] 
        for name in evaluation_metrics:
            if name not in self.METRIC_CLASSES:
                raise KeyError(f"Unknown metric {name}")
            if name == "token_f1":
                self.metrics.append(self.METRIC_CLASSES[name](self))
    
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

    def evaluate(self, pred: Any, gt: Any) -> Dict[str, Any]:
        if not is_not_null(pred) or not is_not_null(gt):
            raise ValueError("Both pred and gt must contain non-null values")
        
        if not isinstance(pred, (list, pd.Series , pl.Series)):
            raise ValueError("pred must be a list, pandas Series, or polars Series")
        
        if not isinstance(gt, (list, pd.Series, pl.Series)):
            raise ValueError("gt must be a list, pandas Series, or polars Series")
        
        # converting pl.Series if it's a list or pandas Series
        if isinstance(gt, (list, pd.Series)):
            gt = pl.Series(gt)
        
        if isinstance(pred, pd.Series):
            pred = pl.Series(pred)

        # check if pred and gt have the same length
        if len(pred) != len(gt):
            raise ValueError("pred and gt must have the same length")

        results: Dict[str, Any] = {}
        for metric in self.metrics:
            res = metric.calculate(pred, gt, parallel=self.parallel, n_procs=self.n_procs)
            results[metric.name()] = res
        
        return results

