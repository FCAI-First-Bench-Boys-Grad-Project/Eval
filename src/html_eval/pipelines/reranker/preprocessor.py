"""
Refactored Batch Preprocessor
- Contains BasePreprocessor (fetching, cleaning, chunking helpers), Preprocessor (single-doc), and BatchPreprocessor (parallel batch).
- Integrated HTML cleaning into BasePreprocessor.
- Parallelism: threaded fetch+clean, process pool chunking.
"""

from __future__ import annotations
from typing import Any, Dict, List, Union
import polars as pl
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from html_eval.core.experiment import Experiment
from html_eval.util.html_util import fetch_content, clean_html ,chunk_html_content 
from html_eval.core.types import Sample
from html_eval.configs.pipeline_config import RerankerPreprocessorConfig

def _chunk_worker(args: tuple) -> Dict[str, Any]:
    sample, config, idx = args
    cleaned_text = clean_html(
        html_content=sample.content,
        extra_remove_tags=config.extra_remove_tags,
        strip_attrs=config.strip_attrs,
        strip_links=config.strip_links,
        keep_tags=config.keep_tags,
        use_clean_rag=config.use_clean_rag)
    
    try:
        if not cleaned_text:
            return {'doc_id': idx, 'chunks': [{'chunkid': f"{idx}-err", 'chunkcontent': '[Chunk Worker ERROR] empty content or fetch failed'}]}
        chunks = chunk_html_content(html_content=cleaned_text,
                                    max_tokens=config.chunk_size,
                                    is_clean=config.use_clean_chunker,
                                    attr_cutoff_len=config.attr_cutoff_len)
        
        chunks_list = [{'chunkid': f"{idx}-{i+1}", 'chunkcontent': c} for i, c in enumerate(chunks)]
        return {'doc_id': idx, 'chunks': chunks_list}
    except Exception as e:
        tb = traceback.format_exc()
        err_payload = {
            "doc_id": idx,
            "chunks": [
                {
                    "chunkid": f"{idx}-err",
                    "chunkcontent": f"[ERROR {type(e).__name__}] {e}\n{tb}"
                }
            ]
        }
        return idx, err_payload


class BasePreprocessor:
    
    def __init__(self,config:RerankerPreprocessorConfig):
        self.config : RerankerPreprocessorConfig = config
        self.fetch_workers : int = config.fetch_workers
        self.cpu_workers : int = config.cpu_workers

        self.extra_remove_tags : List[str] = config.extra_remove_tags
        self.strip_attrs : bool = config.strip_attrs
        self.strip_links : bool = config.strip_links
        self.keep_tags : bool = config.keep_tags
        self.use_clean_rag : bool = config.use_clean_rag
        self.use_clean_chunker : bool = config.use_clean_chunker

        self.chunk_size : int = config.chunk_size
        self.attr_cutoff_len : int = config.attr_cutoff_len
        
        
    def set_experiment(self, experiment: Experiment) -> None:
        """
        Set the experiment instance for logging or other purposes.
        """
        self.experiment = experiment


    def process(self,batch:List[Sample],) -> Union[List[Sample],pl.DataFrame]:
        

        n = len(batch)
        if n == 0:
            return pl.DataFrame([])

        def _quick_fetch(sample: Sample) -> Sample:
            if sample.is_content_url:
                try:
                    fetched = fetch_content(sample.content)
                    sample.content = fetched
                    sample.is_content_url = False
                    return sample
                except Exception as e:
                    sample.content = f"[Fetch ERROR] {e}"
                    sample.is_content_url = False
                    return sample
            else:
                return sample
        # Fetching the content if there are any URLs
        with ThreadPoolExecutor(max_workers=min(self.fetch_workers, max(1, n))) as tpool:
            batch = list(tpool.map(_quick_fetch, batch))
            

        results = [None] * n
        # choose executor class and max_workers
        use_processes = bool(self.cpu_workers and self.cpu_workers > 1)
        ExecutorCls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        if use_processes:
            max_workers = min(self.cpu_workers or n, max(1, n))
        else:
            # for IO-bound thread fallback, use fetch_workers if provided else at most n
            max_workers = min(self.fetch_workers or n, max(1, n))

        # prepare enumerated args (Sample, config, index)
        items = [(batch[i],self.config,i) for i in range(n)]

        results: List[Dict] = [None] * n

        with ExecutorCls(max_workers=max_workers) as ex:
            for idx, res in enumerate(ex.map(_chunk_worker, items)):
                results[idx] = res


        batch_df = pl.DataFrame(batch)
        print("Preprocessed batch :", batch_df)
        print("Results :",results)
        print("With results :",batch_df.hstack(pl.DataFrame(results)))
        return batch_df.hstack(pl.DataFrame(results))


        