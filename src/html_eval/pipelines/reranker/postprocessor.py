from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Iterable, List, Optional, Dict, Any, Union
import os
import polars as pl

from html_eval.util.json_util import extract_and_repair_json , is_schema
from html_eval.util.html_util import find_closest_html_node ,normalize_html_text
from html_eval.core.types import SamplePrediction
from html_eval.configs.pipeline_config import RerankerPostprocessorConfig


# module-level so ProcessPoolExecutor can pickle it
def _safe_extract(response: str, meta: Dict[str, Any], extract_exact: bool) -> Dict[str, Any]:
    try:
        
        parsed_response = extract_and_repair_json(response ,not is_schema(meta['query']))

        if isinstance(parsed_response,str):
            return parsed_response
            
        # validate we have a mapping/dict
        if not isinstance(parsed_response, dict):
            return {"__error__": "[PARSE_ERROR] expected JSON object (dict) from extract_and_repair_json"}
        

        if extract_exact:
            # use .get() and guard when content is missing or empty
            content = meta.get('content') if isinstance(meta, dict) else None
            if not content:
                return {"__error__": "[PARSE_ERROR] exact_extraction requested but no content provided in meta"}

            for attribute, value in list(parsed_response.items()):
                # only attempt to match strings (or convert)
                if value is None:
                    parsed_response[attribute] = None
                    continue
                try:
                    # ensure value is a str for matching
                    val_str = str(value)
                    # best_match = find_best_match(content, val_str)
                    best_match = find_closest_html_node(html_text=content,search_text=val_str)
                    # if best_match is None, keep original or set None
                    parsed_response[attribute] = normalize_html_text(best_match['text']) if best_match and 'text' in best_match else None
                except Exception as e:
                    # keep original but annotate error
                    parsed_response[attribute] = {"__error__": f"[MATCH_ERROR] {e}", "original": value}

        return parsed_response
    except Exception as e:
        return {"__error__": f"[PARSE_ERROR] {e}"}


class PostProcessor:
    """Minimal PostProcessor that turns response strings into SamplePrediction objects."""
    def __init__(self, config: RerankerPostprocessorConfig):
        self._config = config
        self._exact_extraction = bool(config.exact_extraction)

    def process(self, response: str, **meta: Any) -> SamplePrediction:
        """Process one response and wrap result in SamplePrediction."""
        parsed = _safe_extract(response=response, meta=meta, extract_exact=self._exact_extraction)
        return SamplePrediction(prediction=parsed, **meta)

    def process_responses(
        self,
        responses: Iterable[str],
        metas: Optional[Iterable[Dict[str, Any]]] = None,
        n_workers: Optional[int] = None,
        use_process: bool = False,
    ) -> List[SamplePrediction]:
        """
        Parse many responses in parallel and return SamplePrediction list.
        - metas: optional per-response dicts (e.g. {'id':..., 'query':..., 'ground_truth':...})
        - use_process: if True uses ProcessPoolExecutor (workers must be picklable)
        """
        responses = list(responses)
        if metas is None:
            metas = [{} for _ in responses]
        else:
            metas = list(metas)

        if len(metas) != len(responses):
            raise ValueError("Length of metas must match length of responses")

        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

        Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor

        # prepare an iterable for the boolean flag with the same length
        extract_flags = [self._exact_extraction] * len(responses)

        with Executor(max_workers=n_workers) as ex:
            # This will yield parsed dicts in order corresponding to responses
            parsed_iter = ex.map(_safe_extract, responses, metas, extract_flags)
            return [SamplePrediction(prediction=parsed, **meta) for parsed, meta in zip(parsed_iter, metas)]

    def process_dataframe(
        self,
        df: pl.DataFrame,
        response_col: str = "response",
        id_col: str = "id",
        query_col: str = "query",
        gt_col: str = "ground_truth",
        content_col: str = "content",
        filtered_html_col: str = "full_content",
        n_workers: Optional[int] = None,
        use_process: bool = False,
        return_polars: bool = False,
    ) -> Union[List[SamplePrediction], pl.DataFrame]:
        """
        Parse the responses in `df[response_col]`.
        Returns either a list of SamplePrediction (default) or the input DataFrame with a new
        'prediction' column (if return_polars=True).
        """
        if response_col not in df.columns:
            raise KeyError(f"response_col '{response_col}' not present")

        responses = df[response_col].to_list()

        # Use to_dicts() which returns list[dict] of rows and is robust across Polars versions
        rows = df.to_dicts()
        metas: List[Dict[str, Any]] = []
        for row in rows:
            # ensure a string (or empty) content is supplied for exact extraction
            content_val = row.get(content_col)
            metas.append({
                "id": row.get(id_col),
                "query": row.get(query_col),
                "ground_truth": row.get(gt_col),
                "filtered_html": row.get(filtered_html_col),
                "content": content_val if (self._exact_extraction and content_val is not None) else "",
            })

        preds = self.process_responses(
            responses,
            metas=metas,
            n_workers=n_workers,
            use_process=use_process,
        )

        if not return_polars:
            return preds

        pred_values = [p.prediction for p in preds]
        return df.with_columns(pl.Series("prediction", pred_values))
