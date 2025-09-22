import trafilatura
import polars as pl
import multiprocessing
from bs4 import BeautifulSoup, Comment
from ..reranker.preprocessor import BasePreprocessor
from typing import Union, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

class Preprocessor(BasePreprocessor):
    def _preprocess_single(self, html:str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.find_all(['style', 'script', 'head', 'img', 'base', 'noscript']):
            tag.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        content:str = str(soup)

        chunks = self.chunk_content(html=content, max_tokens=500, is_clean_html=True, attr_cutoff_len=50)

        cleaned = [trafilatura.extract(chunk) for chunk in chunks]
        cleaned = [chunk for chunk in cleaned if chunk is not None]
        
        combined_text = ""
        for chunk in cleaned:
            if chunk is None:
                continue
            combined_text += chunk + "\n"
        
        return combined_text

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
                    return_polars: bool = True) -> Union[List[Dict[str, Any]], pl.DataFrame]:
    
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
            jobs = {}

            # Submit fetch tasks for URLs
            for idx in range(n):
                if not is_urls[idx]:
                    html_content[idx] = contents[idx]
                    continue
                else:
                    jobs[tpool.submit(self._fetch_content, contents[idx])] = idx

            # Collect results
            for job in as_completed(jobs):
                idx = jobs[job]
                try:
                    html_content[idx] = job.result()
                except Exception as e:
                    html_content[idx] = f"[FETCH_ERROR] {e}"

        results: List[Dict[str, Any]] = [None] * n
        # chunk_tasks = [(html_content[i] or '', self.chunk_size, self.attr_cutoff_len, doc_id_start + i, queries[i]) for i in range(n)]

        # Use all available CPU cores
        with ProcessPoolExecutor(max_workers=min(self.cpu_workers, n)) as executor:
            # results = list(executor.map(self._preprocess_single, contents))
            jobs = {executor.submit(self._preprocess_single, args): idx for idx, args in enumerate(contents)}
            tmp = {}
            for job in as_completed(jobs):
                idx = jobs[job]
                try:
                    # FIXME DRY Principal
                    result = job.result()
                    tmp[idx] = {'doc_id': doc_id_start + idx,
                                'content': {result},
                                'query': queries[idx]}
                except Exception as e:
                    tmp[idx] = {'doc_id': doc_id_start + idx,
                                'content': f"[Process ERROR] {e}",
                                'query': queries[idx]}
            for i in range(n):
                results[i] = tmp[i]

        if return_polars:
            return pl.DataFrame(results)
        return results