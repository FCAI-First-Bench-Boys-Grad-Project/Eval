"""
Refactored Batch Preprocessor
- Contains BasePreprocessor (fetching, cleaning, chunking helpers), Preprocessor (single-doc), and BatchPreprocessor (parallel batch).
- Integrated HTML cleaning into BasePreprocessor.
- Parallelism: threaded fetch+clean, process pool chunking.
"""

from __future__ import annotations
import re
import requests
from bs4 import BeautifulSoup, Comment
from typing import Any, Dict, List, Optional, Union, Tuple
import polars as pl
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os
from htmlrag import clean_html
from html_chunking import get_html_chunks
from eval.experiment import Experiment

class BasePreprocessor:
    DEFAULT_REMOVE_TAGS = ("script", "style")

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.keep_tags = bool(self.config.get("keep_tags", True))
        self.strip_attrs = bool(self.config.get("strip_attrs", True))
        self.strip_links = bool(self.config.get("strip_links", True))
        self.extra_remove_tags = list(self.config.get("extra_remove_tags", ["header", "footer"]))
        self.timeout = int(self.config.get("timeout", 15))

        self.experiment = None 
    
    def set_experiment(self, experiment: Experiment) -> None:
        """
        Set the experiment instance for logging or other purposes.
        """
        self.experiment = experiment

    def _fetch_content(self, url: str) -> str:
        headers = {
            "User-Agent": self.config.get(
                "user_agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            r = requests.get(url, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            return f"[FETCH_ERROR] {e}"

    def _clean_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content or "", "html.parser")

        remove_tags = set(self.DEFAULT_REMOVE_TAGS) | set(self.extra_remove_tags)
        for tag_name in remove_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        if self.strip_attrs:
            for tag in soup.find_all(True):
                tag.attrs = {}

        if self.strip_links:
            for a in soup.find_all('a'):
                a.replace_with(a.get_text())

        for tag in soup.find_all(True):
            if not tag.get_text(strip=True):
                tag.decompose()

        if self.keep_tags:
            html_str = str(soup)
            html_str = re.sub(r'(?m)^[ \t]*\n', '', html_str)
            return html_str.strip()

        text = soup.get_text(separator='\n', strip=True)
        lines = [line for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        # Using htmlrag cleaning for extra measures
        clean_text = clean_html(clean_text)
        return clean_text.strip()

    def chunk_content(self, content: str, max_tokens: int = 500, is_clean: bool = True, attr_cutoff_len: int = 5) -> List[str]:
        if not content:
            return []
        return get_html_chunks(html=content, max_tokens=max_tokens, is_clean_html=is_clean, attr_cutoff_len=attr_cutoff_len)



def _chunk_worker(args: Tuple[str, int, int, int, str]):
    html_content, max_tokens, attr_cutoff_len, doc_id, query = args
    cleaned_text = BasePreprocessor()._clean_html(html_content) #TODO: I might need to change this in the future , cause it instanciates BasePreprocessor every time
    try:
        if not cleaned_text:
            return {'doc_id': doc_id, 'content': {'chunks': [{'chunkid': f"{doc_id}-err", 'chunkcontent': '[Chunk Worker ERROR] empty content or fetch failed'}]}, 'query': query}
        chunks = get_html_chunks(html=cleaned_text, max_tokens=max_tokens, is_clean_html=True, attr_cutoff_len=attr_cutoff_len)
        chunks_list = [{'chunkid': f"{doc_id}-{i+1}", 'chunkcontent': c} for i, c in enumerate(chunks)]
        return {'doc_id': doc_id, 'content': {'chunks': chunks_list}, 'query': query}
    except Exception as e:
        return {'doc_id': doc_id, 'content': {'chunks': [{'chunkid': f"{doc_id}-err", 'chunkcontent': f"[Chunk Worker ERROR] {e}"}]}, 'query': query}


class Preprocessor(BasePreprocessor):
    """
    Preprocessor for cleaning, fetching, and chunking HTML or text documents.

    This class extends `BasePreprocessor` and provides functionality for:
      - Accepting a batch of documents (as Polars DataFrame, list of dicts, or dict of lists).
      - Fetching HTML content if documents are URLs.
      - Cleaning and chunking HTML/text content into smaller segments for downstream tasks.
      - Supporting both multi-threaded fetching and multi-processing for CPU-bound chunking.

    Parameters
    ----------
    chunk_size : int, optional (default=500)
        Maximum number of tokens per chunk when splitting documents.
    attr_cutoff_len : int, optional (default=5)
        Minimum length of text attributes to retain when processing HTML.
    preprocessor_cfg : dict, optional
        Additional configuration dictionary passed to the `BasePreprocessor`.
    fetch_workers : int, optional
        Number of threads used for fetching URLs. Defaults to:
        `min(32, max(4, (os.cpu_count() or 2) * 2))`.
    cpu_workers : int, optional
        Number of processes used for CPU-intensive chunking. Defaults to
        `(os.cpu_count() - 1)` if available, else 1.

    Methods
    -------
    _ensure_polars(df_like):
        Ensures the input is converted into a Polars DataFrame.
    process_batch(batch, content_col='content', query_col='query',
                  is_url_col=None, doc_id_start=1, return_polars=False):
        Processes a batch of documents:
          - Fetches content for URLs (multi-threaded).
          - Cleans and chunks content (multi-process or multi-thread).
          - Returns results as list of dicts or Polars DataFrame.
    """
    
    def __init__(self, *, chunk_size: int = 500, attr_cutoff_len: int = 5, preprocessor_cfg: Optional[Dict[str, Any]] = None,
                 fetch_workers: Optional[int] = None, cpu_workers: Optional[int] = None) -> None:
        super().__init__(preprocessor_cfg)
        self.chunk_size = int(chunk_size)
        self.attr_cutoff_len = int(attr_cutoff_len)
        self.fetch_workers = fetch_workers if fetch_workers is not None else min(32, max(4, (os.cpu_count() or 2) * 2))
        default_cpu = max(1, (os.cpu_count() or 2) - 1)
        self.cpu_workers = cpu_workers if cpu_workers is not None else default_cpu

    def _ensure_polars(self, df_like: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, List[Any]]]) -> pl.DataFrame:
        if isinstance(df_like, pl.DataFrame):
            return df_like
        return pl.DataFrame(df_like)

    def process_batch(self,
                      batch: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, List[Any]]],
                      content_col: str = 'content',
                      query_col: str = 'query',
                      is_url_col: Optional[str] = None, # Optional column to indicate if content is a URL
                      doc_id_start: int = 1,
                      return_polars: bool = False) -> Union[List[Dict[str, Any]], pl.DataFrame]:
        
        df = self._ensure_polars(batch)

        if content_col not in df.columns or query_col not in df.columns:
            raise ValueError(f"Input batch must have columns '{content_col}' and '{query_col}'")

        contents = df[content_col].to_list()
        queries = df[query_col].to_list()

        # Check if is_url_col is provided and if not check whether the contents are URLs
        if is_url_col and is_url_col in df.columns:
            is_urls = df[is_url_col].to_list()
        else:
            is_urls = [bool(str(h).lower().startswith(('http://', 'https://'))) for h in contents]

        n = len(contents)
        if n == 0:
            return [] if not return_polars else pl.DataFrame([])

        # Fetching the content if there are any URLs
        html_content: List[Optional[str]] = [None] * n
        with ThreadPoolExecutor(max_workers=min(self.fetch_workers, max(1, n))) as tpool:
            futs = {}

            # Submit fetch tasks for URLs
            for idx in range(n):
                if not is_urls[idx]:
                    html_content[idx] = contents[idx]
                    continue
                else:
                    futs[tpool.submit(self._fetch_content, contents[idx])] = idx

            # Collect results
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    html_content[idx] = fut.result()
                except Exception as e:
                    html_content[idx] = f"[FETCH_ERROR] {e}"

        results: List[Dict[str, Any]] = [None] * n
        chunk_tasks = [(html_content[i] or '', self.chunk_size, self.attr_cutoff_len, doc_id_start + i, queries[i]) for i in range(n)]

        # If we have more than one CPU core available, use ProcessPoolExecutor for chunking (allows for true parallelism)
        if self.cpu_workers and self.cpu_workers > 1:

            with ProcessPoolExecutor(max_workers=min(self.cpu_workers, n)) as pex:
                futs = {pex.submit(_chunk_worker, args): idx for idx, args in enumerate(chunk_tasks)}
                tmp = {}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        tmp[idx] = fut.result()
                    except Exception as e:
                        doc_id = doc_id_start + idx
                        tmp[idx] = {'doc_id': doc_id, 'content': {'chunks': [{'chunkid': f"{doc_id}-err", 'chunkcontent': f"[Process ERROR] {e}"}]}, 'query': queries[idx]}
                for i in range(n):
                    results[i] = tmp[i]
        # If we have only one CPU core or no CPU workers specified, use ThreadPoolExecutor for chunking
        else:
            with ThreadPoolExecutor(max_workers=min(self.fetch_workers, n)) as tpool2:
                futs = {tpool2.submit(_chunk_worker, args): idx for idx, args in enumerate(chunk_tasks)}
                tmp = {}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        tmp[idx] = fut.result()
                    except Exception as e:
                        doc_id = doc_id_start + idx
                        tmp[idx] = {'doc_id': doc_id, 'content': {'chunks': [{'chunkid': f"{doc_id}-err", 'chunkcontent': f"[Thread ERROR] {e}"}]}, 'query': queries[idx]}
                for i in range(n):
                    results[i] = tmp[i]

        if return_polars:
            return pl.DataFrame(results)
        return results


# Example (not executed here):
# df = pl.DataFrame({'html': ['https://example.com','<html><body>Hello</body></html>'], 'query':['q1','q2']})
# bp = BatchPreprocessor(chunk_size=400, fetch_workers=8, cpu_workers=4)
# out = bp.process_batch(df)
# print(out)
