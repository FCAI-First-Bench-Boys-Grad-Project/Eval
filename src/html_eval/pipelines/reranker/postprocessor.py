from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Iterable, List, Optional, Dict, Any, Union
import os
import polars as pl

from html_eval.util.json_util import extract_and_repair_json
from html_eval.core.types import SamplePrediction


# module-level so ProcessPoolExecutor can pickle it
def _safe_extract(response: str) -> Dict[str, Any]:
    try:
        return extract_and_repair_json(response)
    except Exception as e:
        return {"__error__": f"[PARSE_ERROR] {e}"}


class PostProcessor:
    """Minimal PostProcessor that turns response strings into SamplePrediction objects."""

    def process(self, response: str, **meta: Any) -> SamplePrediction:
        """Process one response and wrap result in SamplePrediction."""
        parsed = _safe_extract(response)
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

        with Executor(max_workers=n_workers) as ex:
            parsed_iter = ex.map(_safe_extract, responses)
            return [SamplePrediction(prediction=parsed, **meta) for parsed, meta in zip(parsed_iter, metas)]

    def process_dataframe(
        self,
        df: pl.DataFrame,
        response_col: str = "response",
        id_col: str = "id",
        query_col: str = "query",
        gt_col: str = "ground_truth",
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
        metas = []
        # collect minimal meta fields if present
        for row in df.iter_rows(named=True):
            metas.append({
                "id": row.get(id_col),
                "query": row.get(query_col),
                "ground_truth": row.get(gt_col),
            })

        preds = self.process_responses(
            responses,
            metas=metas,
            n_workers=n_workers,
            use_process=use_process,
        )

        if not return_polars:
            return preds

        # add 'prediction' column to DataFrame (preserves order)
        pred_values = [p.prediction for p in preds]
        return df.with_columns(pl.Series("prediction", pred_values))
