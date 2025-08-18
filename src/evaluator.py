from abc import ABC, abstractmethod
from typing import List, Dict, Any
from numpy.typing import NDArray
import numpy as np
from src.html_datasets.base import BaseHTMLDataset
import polars as pl
from jsonschema import validate, ValidationError
import ast
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from typing import Any, List, Dict, Union, Optional


class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def calculate(self, pred: pl.DataFrame , gt: pl.DataFrame) -> float:
        """
        Calculate the metric value.
        
        Args:
            ground_truth: Ground truth output
            predictions: Predicted output
            
        Returns:
            float: Calculated metric value
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""
        pass

class Evaluator:
    """Class for evaluating predictions using multiple metrics."""
    METRIC_REGISTRY = {
        
    }

    def __init__(self, dataset: BaseHTMLDataset):
        """
        Initialize the evaluator with a list of metrics.
        
        Args:
            metrics: List of Metric objects to use for evaluation
        """
        self.dataset = dataset
        self.metrics = [self.METRIC_REGISTRY[metric]() for metric in dataset.evaluation_metrics]
    
    def evaluate(self, pred: pl.DataFrame , gt: pl.DataFrame) -> Dict[str, float]:
        """
        Evaluate predictions using all metrics.
        
        Args:
            ground_truth: Ground truth output
            predictions: Predicted output
            
        Returns:
            Dict mapping metric names to their calculated values
        """
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.calculate(pred, gt)
        return results
    

class RecallJson(Metric):
    """The proportion of all relevant key information that is successfully extracted. Performing Macro averaging."""
    
    def calculate(self, ground_truth: NDArray[Any], predictions: NDArray[Any]) -> float:
        '''
        The input should be a numpy array of dictionaries.
        '''

        true_positive = 0
        # Loop through each dictionary in the ground truth and check if it is in the predictions
        for i in range(len(ground_truth)):
            sample_true_positive = 0
            for key , value in ground_truth[i].items():
                if key in predictions[i].keys() and value == predictions[i][key]:
                    sample_true_positive += 1
            true_positive += sample_true_positive / len(ground_truth[i])

        return true_positive / len(ground_truth)

                    
    
    @property
    def name(self) -> str:
        return "recall"
    
class PrecisionJson(Metric):
    """The proportion of all extracted key information that is relevant. Performing Macro averaging."""
    
    def calculate(self, ground_truth: NDArray[Any], predictions: NDArray[Any]) -> float:
        '''
        The input should be a numpy array of dictionaries.
        '''
        true_positive = 0
        for i in range(len(ground_truth)):
            sample_true_positive = 0
            for key , value in ground_truth[i].items():
                if key in predictions[i].keys() and value == predictions[i][key]:
                    sample_true_positive += 1
            true_positive += sample_true_positive / len(predictions[i])
        return true_positive / len(predictions)
                
    @property
    def name(self) -> str:
        return "precision"

class F1Json(Metric):
    """The harmonic mean of precision and recall. Performing Macro averaging."""
    
    def calculate(self, ground_truth: NDArray[Any], predictions: NDArray[Any]) -> float:
        '''
        The input should be a numpy array of dictionaries.
        '''
        precision = PrecisionJson().calculate(ground_truth, predictions)
        recall = RecallJson().calculate(ground_truth, predictions)
        return 2 * (precision * recall) / (precision + recall)
    
    @property
    def name(self) -> str:
        return "f1"



# --- Utility Function ---
def is_not_null(val: Any) -> bool:
    """
    Checks if a value is not null or NaN.
    Handles scalars, lists, numpy arrays, and pandas Series.
    For array-like structures, returns False if all elements are NA or the structure is empty.
    """
    if val is None:
        return False

    if isinstance(val, pd.Series):
        if val.empty: # Explicitly check if the Series is empty
            return False
        return not val.isna().all() # Use Series' own isna() method

    if isinstance(val, np.ndarray):
        if val.size == 0: # For numpy arrays, check .size
            return False
        return not pd.isna(val).all() # pd.isna works well on np.ndarray

    if isinstance(val, pl.DataFrame):
        if val.is_empty():
            return False
        # For Polars DataFrame, check if all values are null
        return not val.select(pl.all().is_null()).all().item()  # Returns True

    if isinstance(val, list):
        if not val:  # Empty list
            return False
        # For a list, determine if it contains any non-null values.
        # Converting to a pandas Series with dtype=object is a robust way
        # to handle mixed types and various NA representations (None, np.nan).
        try:
            return not pd.Series(val, dtype='object').isna().all()
        except Exception:
            # Fallback for rare cases where Series creation might fail or for extreme mixed types.
            # A simpler check: is there at least one item that is not None?
            return any(item is not None for item in val)

    # For scalars or other types pd.isna can handle directly
    return not pd.isna(val)

# --- Core Matching Logic ---
class Matcher:
    """
    Handles the comparison logic between a ground truth value and a predicted value.
    It supports both exact and fuzzy matching.
    """

    def __init__(self, is_fuzzy: bool = False, fuzzy_threshold: int = 90):
        """
        Initializes the Matcher.

        Args:
            is_fuzzy (bool): If True, use fuzzy matching. Otherwise, use exact matching.
            fuzzy_threshold (int): The minimum similarity score (0-100) for a fuzzy match
                                   to be considered a match.
        """
        if not (0 <= fuzzy_threshold <= 100):
            raise ValueError("fuzzy_threshold must be between 0 and 100.")
        self.is_fuzzy = is_fuzzy
        self.fuzzy_threshold = fuzzy_threshold

    def _normalize_ground_truth(self, gt_val: Any) -> List[Any]:
        """
        Normalizes a ground truth value into a list of non-null items.
        - Handles None: returns empty list.
        - Handles strings that might represent lists (e.g., "['a', 'b']").
        - Ensures single values are wrapped in a list.
        - Filters out None/NaN values from the resulting list using `is_not_null` on items.
        """
        # Initial check for overall nullity of gt_val itself
        if not is_not_null(gt_val) and not isinstance(gt_val, list): # if scalar like None, np.nan
             if not (isinstance(gt_val, str) and gt_val.strip().startswith('[')): # unless it's a string list
                return []
        elif isinstance(gt_val, list) and not any(is_not_null(item) for item in gt_val): # empty list or list of all nulls
            return []


        gt_list: List[Any]
        if isinstance(gt_val, str):
            try:
                # Attempt to evaluate string as a Python literal (e.g., list, number, string)
                evaluated = ast.literal_eval(gt_val)
                if isinstance(evaluated, list):
                    gt_list = evaluated
                else:
                    # It was a literal but not a list (e.g. "123", "'text'")
                    gt_list = [evaluated]
            except (ValueError, SyntaxError):
                # Not a valid Python literal, treat as a single string element
                gt_list = [gt_val]
        elif not isinstance(gt_val, list):
            gt_list = [gt_val] # It's a single non-list item
        else:
            gt_list = gt_val # It's already a list

        # Filter out any None or NaN items from the list itself
        # is_not_null here applies to individual items, which should be scalars mostly
        return [item for item in gt_list if is_not_null(item)]


    def compare(self, gt_val: Any, pred_val: Any) -> float:
        """
        Compares a predicted value against a ground truth value (or list of values).

        Args:
            gt_val: The ground truth value. Can be a single value, a list,
                    or a string representation of a list.
            pred_val: The predicted value.

        Returns:
            float: 1.0 if a match is found, 0.0 otherwise.
        """
        if not is_not_null(pred_val):
            return 0.0  # Prediction is null/NaN, cannot match

        normalized_gt_list = self._normalize_ground_truth(gt_val)
        if not normalized_gt_list: # No valid (non-null) ground truth values to match against
            return 0.0

        pred_str = str(pred_val) # Common string representation for prediction

        if self.is_fuzzy:
            max_score = 0
            for gt_item in normalized_gt_list:
                try:
                    gt_item_str = str(gt_item)
                    score = fuzz.ratio(gt_item_str.lower(), pred_str.lower()) # Optional: make fuzzy case-insensitive
                    if score > max_score:
                        max_score = score
                except Exception as e:
                    print(f"Warning: Error during fuzzy match (GT: {gt_item}, Pred: {pred_val}): {e}")
            return 1.0 if max_score >= self.fuzzy_threshold else 0.0
        else: # Exact match
            for gt_item in normalized_gt_list:
                # Case 1: Both are numeric (int/float) - direct comparison
                if isinstance(pred_val, (int, float)) and isinstance(gt_item, (int, float)):
                    # np.isclose can be used for float comparisons if tolerance is needed
                    if pred_val == gt_item:
                        return 1.0
                # Case 2: General comparison using string forms
                # This handles "text" == "text", 1 == "1" (after str conversion), etc.
                # For exact match, consider if case sensitivity is desired.
                # Current: str(gt_item) == pred_str (case-sensitive if strings)
                elif str(gt_item) == pred_str: # Direct string comparison
                    return 1.0
                # Alternative for case-insensitive exact match:
                # elif str(gt_item).lower() == pred_str.lower():
                #     return 1.0
            return 0.0

# --- Row-Level Evaluation Metrics ---
class RowEvaluator:
    """
    Calculates row-level precision, recall, and F1-score using a specified Matcher.
    Assumes rows are dictionaries where keys are attribute names and values are attribute values.
    """

    def __init__(self, matcher: Matcher):
        """
        Initializes the RowEvaluator with a Matcher instance.

        Args:
            matcher (Matcher): The matcher object to use for comparing attribute values.
        """
        self.matcher = matcher

    def _clean_row_data(self, row_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensures row data is a dictionary, returning an empty dict if None."""
        return row_data if row_data is not None else {}

    def recall(self, gt_row: Optional[Dict[str, Any]], pred_row: Optional[Dict[str, Any]]) -> float:
        """
        Calculates recall for a single row.
        Recall = TP / (Number of actual positive attributes in GT)
        Ignores attributes in GT that are null/NaN (as determined by is_not_null).
        """
        gt_row_clean = self._clean_row_data(gt_row)
        pred_row_clean = self._clean_row_data(pred_row)

        true_positives = 0
        gt_attributes_count = 0

        if not gt_row_clean:
            return 0.0

        for attribute, gt_value in gt_row_clean.items():
            # Skip attribute if ground truth value is considered null
            if not is_not_null(gt_value):
                continue

            gt_attributes_count += 1
            predicted_value = pred_row_clean.get(attribute)
            true_positives += self.matcher.compare(gt_value, predicted_value)

        return true_positives / gt_attributes_count if gt_attributes_count > 0 else 0.0

    def precision(self, gt_row: Optional[Dict[str, Any]], pred_row: Optional[Dict[str, Any]]) -> float:
        """
        Calculates precision for a single row.
        Precision = TP / (Number of predicted positive attributes)
        A predicted attribute is "positive" if its value is not null/NaN (as per is_not_null).
        TPs are counted for attributes present in the ground truth row.
        """
        gt_row_clean = self._clean_row_data(gt_row)
        pred_row_clean = self._clean_row_data(pred_row)

        true_positives = 0
        predicted_positives_count = 0

        if not pred_row_clean:
            return 0.0

        # Count all non-null predicted attributes for the denominator
        for pred_value_for_count in pred_row_clean.values():
            if is_not_null(pred_value_for_count):
                predicted_positives_count += 1

        if predicted_positives_count == 0:
            return 0.0 # No positive predictions, so precision is 0

        # Calculate true positives based on matches for attributes in the GT row
        for attribute, gt_value_for_tp in gt_row_clean.items():
            predicted_value = pred_row_clean.get(attribute)
            # Only attempt to match if the prediction for this attribute is non-null
            if is_not_null(predicted_value):
                # gt_value_for_tp can be null here; matcher.compare handles null GT values.
                true_positives += self.matcher.compare(gt_value_for_tp, predicted_value)

        return true_positives / predicted_positives_count if predicted_positives_count > 0 else 0.0


    def f1_score(self, gt_row: Optional[Dict[str, Any]], pred_row: Optional[Dict[str, Any]]) -> float:
        """
        Calculates the F1 score for a single row.
        """
        p = self.precision(gt_row, pred_row)
        r = self.recall(gt_row, pred_row)

        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
