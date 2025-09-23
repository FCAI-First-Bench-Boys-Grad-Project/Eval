from __future__ import annotations
import re
import string
import ast
from typing import Any, List, Tuple
from collections import Counter
import pandas as pd
import polars as pl
from json_repair import repair_json
import json


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
def squad_normalize_answer(s: str) -> str:
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

def squad_get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return squad_normalize_answer(s).split()

def compute_f1_squad(a_gold: str, a_pred: str) -> Tuple[float, float, float]:
    """returns f1, precision, recall for SQuAD-style answers"""
    gold_toks = squad_get_tokens("" if a_gold is None else str(a_gold))
    pred_toks = squad_get_tokens("" if a_pred is None else str(a_pred))
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

def repair_and_parse(s: str) -> dict:
    """Robust parse: try json.loads, then repair_json->loads, then ast.literal_eval, else {}."""
    if s is None:
        return {}
    # Fast attempt
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try json_repair then parse
    try:
        repaired = repair_json(s)
        return json.loads(repaired)
    except Exception:
        pass

    # Last resort: python literal eval for python-style dict strings
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    # Give up
    return {}
