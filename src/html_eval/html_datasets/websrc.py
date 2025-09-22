from html_eval.html_datasets.base import BaseHTMLDataset , Sample
from html_eval.configs.dataset_config import WebSrcConfig
import polars as pl

class WebSrcDataset(BaseHTMLDataset):
    """
    Dataset class for handling web source data in HTML format.
    Inherits from BaseHTMLDataset to provide basic functionality.
    To initialize this class you need two jsonl files:
    - one with the HTML content with the following fields:
        - `id`: unique identifier for the page
        - `website`: the website from which the page was scraped
        - `html`: the HTML content of the page
        - `domain`: the domain of the page
    - another with the queries and ground truth with the following fields:
        - `id`: unique identifier for the query
        - `question`: the query text
        - `answer`: the ground truth answer
        - `element_id`: the id of the HTML element that contains the answer
        - `answer_start`: the start index of the answer in the HTML content
    """
    def __init__(self, config: WebSrcConfig):

        super().__init__(config=config)
        self.html_source_path = config.html_source_path
        self.data_source_path = config.data_source_path
        
        self.html_content_df = pl.read_ndjson(self.html_source_path)
        self.data_df = pl.read_ndjson(self.data_source_path)
        
        self.abbreviation_to_domain = {row['domain'][:2]: row['domain'] for row in self.html_content_df.to_dicts()}
        
    
    def _get_total_length(self) -> int:
        """Return number of samples"""
        return len(self.data_df)

    
    def _get_item(self, idx: int) -> Sample:
        """Return (html, query, ground_truth) tuple for index"""
        if idx < 0 or idx >= self._get_total_length():
            raise IndexError("Index out of bounds")
        
        # Access the row as a dictionary for easier field access
        row = self.data_df[idx].to_dicts()[0]
        
        domain_abb = row['id'][:2]
        domain = self.abbreviation_to_domain.get(domain_abb)
        website_id = int(row['id'][2:9])

        html_row_df = self.html_content_df.filter(
            (pl.col('domain') == domain) & 
            (pl.col('id').cast(pl.Int32) == website_id)
        )
        
        return Sample({
            "id": row['id'],
            "html_content": html_row_df['html'][0] if not html_row_df.is_empty() else None,
            "query": row['question'],
            "ground_truth": row['answer']
        })

    