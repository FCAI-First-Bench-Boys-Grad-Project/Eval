from src.html_datasets.base import BaseHTMLDataset
from typing import Iterator, List, Optional, Tuple, Any
import pandas as pd
import random

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
    def __init__(self, html_source_path: str, data_source_path: str):
        super().__init__()
        self.html_source_path = html_source_path
        self.data_source_path = data_source_path
        self.html_content_df = pd.read_json(html_source_path, lines=True)
        self.data_df = pd.read_json(data_source_path, lines=True)
        # Initialize other necessary attributes, e.g., loading data from source
        self.abb_to_domain = {row['domain'][:2]: row['domain'] for _, row in self.html_content_df.iterrows()}
        
        # it needs to match the names used in the evaluation script
        self.evaluation_metrics = ['exact_match', 'f1_token_level']  # Example metrics, adjust as needed
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.data_df)

    
    def __getitem__(self, idx: int) -> Tuple[Optional[str], Optional[str], Any]:
        """Return (html, query, ground_truth) tuple for index"""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        
        row = self.data_df.iloc[idx]
        domain = self.abb_to_domain[row['id'][:2]]
        website_id = row['id'][2:9]
        
        html_row = self.html_content_df[
            (self.html_content_df['domain'] == domain) & 
            (self.html_content_df['id'] == int(website_id))
        ]
        
        html = html_row['html'].iloc[0] if not html_row.empty else None
        query = row['question']
        ground_truth = {
            'answer': row['answer'],
            'element_id': row['element_id'],
            'answer_start': row['answer_start']
        }
        
        return html, query, ground_truth

    
    def __iter__(self) -> Iterator[Tuple[Optional[str], Optional[str], Any]]:
        """Iterate over (html, query, ground_truth) tuples"""
        for idx in range(len(self)):
            yield self[idx]


    def batch_iterator(
        self, batch_size: int, shuffle: bool = False
    ) -> Iterator[List[Tuple[Optional[str], Optional[str], Any]]]:
        """Iterate over batches of (html, query, ground_truth) tuples"""
        if shuffle:
            indices = list(range(len(self)))
            random.shuffle(indices)
        else:
            indices = range(len(self))

        batch = []
        for idx in indices:
            batch.append(self[idx])
            if len(batch) == batch_size:
                yield batch
                batch = []

        # Yield the last batch if it has leftover samples
        if batch:
            yield batch