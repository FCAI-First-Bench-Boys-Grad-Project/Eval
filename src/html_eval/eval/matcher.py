# master_evaluator.py
from __future__ import annotations
import re
import ast
from dataclasses import dataclass
from typing import Any, List, Optional
import html
import textwrap
from html_eval.util.eval_util import is_not_null
from html_eval.util.html_util import normalize_text

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
        Normalization includes:
        - HTML unescaping (&amp; -> &)
        - Dedenting multi-line text
        - Stripping leading/trailing whitespace
        """
        def clean_text(x: str) -> str:
            if not isinstance(x, str):
                return x
            x = textwrap.dedent(x)              # remove indentation
            x = html.unescape(x)                # decode &amp; etc.
            x = re.sub(r'\s+', ' ', x).strip()  # normalize spaces
            return x

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

        # normalize each item
        return [clean_text(it) for it in items if is_not_null(it)]


    def compare(self, gt: Any, pred: Any) -> bool:
        """
        Return True if pred matches any gt candidate.
        Exact match by default, fuzzy if configured.
        """
        # if not is_not_null(pred):
        #     return False
        candidates = self._normalize_gt(gt)
        if not candidates:
            if not is_not_null(pred):
                return True
            return False
        
        pred_s = normalize_text(str(pred))
        
        if self.cfg.is_fuzzy:
            threshold = max(0, min(100, self.cfg.fuzzy_threshold))
            for c in candidates:
                try:
                    cs = normalize_text(str(c))
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
                if normalize_text(str(c)) == pred_s:
                    return True
            return False

