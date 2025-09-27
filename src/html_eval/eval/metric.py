from __future__ import annotations
from typing import Any, Dict, List , TYPE_CHECKING
from abc import ABC, abstractmethod
import polars as pl
from html_eval.util.eval_util import is_not_null, repair_and_parse, compute_f1_squad
from html_eval.eval.matcher import Matcher, MatcherConfig
from html_eval.core.types import SamplePrediction , SampleEvaluation

if TYPE_CHECKING:
    from html_eval.eval.evaluator import Evaluator

METRICS_REGISTRY: dict[str, type[Metric]] = {}

def register_metric(name: str):
    def wrapper(cls):
        METRICS_REGISTRY[name] = cls
        return cls
    return wrapper

class Metric(ABC):
    def __init__(self,evaluator: "Evaluator"):
        super().__init__()

        
        self._values: Dict[str, Any] = {}
        self._sample_eval: List[SampleEvaluation] = []
        self.evaluator = evaluator  # Reference to the Evaluator instance

    @property
    def values(self) -> Dict[str, Any]:
        """
        Returns the aggregated values of the metric.
        """
        return self._values
    
    @property
    def sample_evaluations(self) -> List[SampleEvaluation]:
        """
        Returns the list of SampleEvaluation objects containing detailed evaluation results.
        """
        return self._sample_eval

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def calculate(self,predictions: List[SamplePrediction]) -> Dict[str, Any]:
       
        ...


@register_metric("token_f1")
class TokenF1(Metric):
    def name(self) -> str:
        return "token_f1"
    
    def _normalize_into_string(self,value: Any) -> str:
        """
        Normalize any input into a string to avoid Polars dtype issues.
        - None -> ""
        - dict/list -> JSON string
        - everything else -> str()
        """
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            import json
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)
    
    def calculate(self,predictions:List[SamplePrediction]) -> Dict[str, Any]:
        """
        Calculate token-level F1 score for SQuAD-style answers.
        """
        # F1 calculation for each pair
        def _calculate_f1(row: tuple[str, str]) -> Dict[str, float]:
            a_pred, a_gold = row
            f1, prec, rec = compute_f1_squad(a_gold, a_pred)
            return {"f1": f1, "precision": prec, "recall": rec}

        # Convert predictions to Polars DataFrame
        predictions = [
            SamplePrediction(
                id=pred.id,
                query=pred.query,
                ground_truth=pred.ground_truth,
                prediction=self._normalize_into_string(pred.prediction)
            ) for pred in predictions
            ]
        
        pred_df = pl.DataFrame(predictions)
        
        rows = zip(pred_df["prediction"].cast(pl.Utf8),pred_df["ground_truth"].cast(pl.Utf8))

        results = [_calculate_f1(row) for row in rows]

        # Aggregate
        f1_scores = [res["f1"] for res in results]
        precision_scores = [res["precision"] for res in results]
        recall_scores = [res["recall"] for res in results]  

        # Save into self._values 
        self._values["f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        self._values["precision"] = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        self._values["recall"] = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

        # Create SampleEvaluation objects
        self._sample_eval = []
        for pred_item , res in zip(predictions,results):
            eval_entry = SampleEvaluation(
                id=pred_item.id,
                query=pred_item.query,
                ground_truth=pred_item.ground_truth,
                prediction=pred_item.prediction,
                evaluation=res
            )
            self._sample_eval.append(eval_entry)

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


@register_metric("page_level_f1")
class PageLevelF1(Metric):
    def name(self) -> str:
        return "page_level_f1"
    
    def calculate(self,predictions: List[SamplePrediction]) -> Dict[str, Any]:
        """
        Calculate page-level F1 score for json answers.
        Implementing both Markuplm and swde evaluation method.
        
        I assume that pred and gt are polars Series containing Dict[str, Any] or similar structures.       
        """
        # Turning the gt Series into a Series of Dict[str, Any]
        gt = [repair_and_parse(pred.ground_truth) for pred in predictions]
        pred = [pred.prediction for pred in predictions]
        schema = gt[0].keys()

        print("Prediction ", pred)
        print("Ground Truth ", gt)
        # Get the schema of the Domain
        
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

        # Collect results per website and per field
        dataset_valid_indices = self.evaluator.experiment.data.get_all_indices()

        print("Prediction Type", type(pred))
        print("Ground Truth Type", type(gt))
        print("Dataset Valid Indices: ", dataset_valid_indices)
        
        for idx, (pred, gt) in enumerate(zip(pred, gt)):
            if not is_not_null(pred) or not is_not_null(gt):
                continue
            og_idx = dataset_valid_indices[idx]  # Map back to original index in dataset
            current_sample_eval = {}
            print(f"Processing index {og_idx}")
            print(self.evaluator.experiment.data.get_website_name(og_idx))
            current_website = self.evaluator.experiment.data.get_website_name(og_idx)
            # Iterate over each field in the schema
            for field in schema:
                try:
                    pred_match_gt = em_matcher.compare(gt[field], pred[field])
                except:
                    print(f"There is an error with field {field}. GT:{gt} and Pred:{pred}")
                    pred_match_gt = False

                current_sample_eval[field] = 1 if pred_match_gt else 0

                # Initialize nested dicts if not present
                if current_website not in results:
                    results[current_website] = {}
                if field not in results[current_website]:
                    results[current_website][field] = {
                        'page_hits': 0,
                        'extracted_pages': 0,
                        'ground_truth_pages': 0,
                        'f1': 0.0,
                        'precision': 0.0,
                        'recall': 0.0
                    }

                if pred_match_gt:
                    results[current_website][field]['page_hits'] += 1
                if is_not_null(pred) and is_not_null(pred[field]):
                    results[current_website][field]['extracted_pages'] += 1
                if is_not_null(gt) and is_not_null(gt[field]):
                    results[current_website][field]['ground_truth_pages'] += 1

            # Create SampleEvaluation object for the current sample
            eval_entry = SampleEvaluation(
                id=predictions[idx].id,
                query=predictions[idx].query,
                ground_truth=predictions[idx].ground_truth,
                prediction=predictions[idx].prediction,
                evaluation=current_sample_eval
            )
            self._sample_eval.append(eval_entry)

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
        self._values['website_aggregated_results'] = website_aggregated_results
        self._values['field_aggregated_results'] = field_aggregated_results
        self._values['results'] = results

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
