# postprocessor_polars.py
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Iterable, List, Optional , Union
import json
from json_repair import repair_json
import polars as pl
import os
from html_eval.experiment import Experiment



def _extract_and_repair_json(response: Union[str,dict]) -> dict:
    """
    Worker function for processing a single LLM response.
    Returns a dict (empty dict on error). Top-level function for pickling.
    """
    if response is None:
        return {}
    
    if isinstance(response, dict):
        return response  # already a dict
    try:
        json_string = response

        # common fenced codeblock ```json ... ```
        if "```json" in response:
            try:
                json_string = response.split("```json", 1)[1].split("```", 1)[0]
            except Exception:
                json_string = response

        # fallback: try to capture the first {...} block
        if "{" in json_string and "}" in json_string:
            start_index = json_string.find("{")
            end_index = json_string.rfind("}") + 1
            json_string = json_string[start_index:end_index]

        repaired = repair_json(json_string)
        parsed = json.loads(repaired)

        print('-'*80)
        print(f"Original Response: {response}")
        print(f"Parsed JSON: {parsed}")
        print('-'*80)

        # Checking if the parsed JSON just contains a single key "answer"
        if isinstance(parsed, dict) and len(parsed) == 1 and "answer" in parsed:
            # If it does, we can return it directly
            return parsed['answer']
        return parsed
    except Exception:
        return {}


class PostProcessor:
    def __init__(self):
        self.experiment = None 
    
    def set_experiment(self, experiment: Experiment) -> None:
        """
        Set the experiment instance for logging or other purposes.
        """
        self.experiment = experiment

    def process(self, response: str) -> dict:
        """Single response, backward-compatible."""
        return _extract_and_repair_json(response)

    def process_responses(
        self,
        responses: Iterable[str],
        n_workers: Optional[int] = None,
        use_process: bool = True,
        chunksize: int = 1,
        show_progress: bool = True,
    ) -> List[dict]:
        """
        Process an iterable of response strings in parallel and return a list of dicts.
        """
        responses = list(responses)
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

        Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor

        use_tqdm = show_progress
        if use_tqdm:
            try:
                from tqdm.auto import tqdm
            except Exception:
                use_tqdm = False

        results: List[dict]
        with Executor(max_workers=n_workers) as ex:
            if use_tqdm:
                it = ex.map(_extract_and_repair_json, responses, chunksize=chunksize)
                results = [r for r in tqdm(it, total=len(responses))]
            else:
                results = list(ex.map(_extract_and_repair_json, responses, chunksize=chunksize))

        return results

    def process_dataframe(
        self,
        df: pl.DataFrame,
        response_col: str = "response",
        result_col: str = "json",
        n_workers: Optional[int] = None,
        use_process: bool = True,
        chunksize: int = 1,
        show_progress: bool = False,
    ) -> pl.DataFrame:
        """
        Process responses in a Polars DataFrame and return a new DataFrame
        with a Struct column containing the parsed JSON.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError("df must be a polars.DataFrame")

        if response_col not in df.columns:
            raise KeyError(f"response_col '{response_col}' not in dataframe columns: {df.columns}")

        responses = df[response_col].to_list()
        dict_results = self.process_responses(
            responses,
            n_workers=n_workers,
            use_process=use_process,
            chunksize=chunksize,
            show_progress=show_progress,
        )

        # Convert list of dicts to a Polars Struct column
        # struct_series = pl.Series(result_col, dict_results).cast(pl.Struct(dict_results[0] if dict_results else {}))
        # df_out = df.with_columns(struct_series)

        # return df_out

        return dict_results

