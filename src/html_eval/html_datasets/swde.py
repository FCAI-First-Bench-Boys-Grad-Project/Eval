from html_eval.html_datasets.base_html_dataset import BaseHTMLDataset
from html_eval.core.types import Sample
from html_eval.configs.dataset_config import SWDEConfig
from typing import List
from datasets import load_dataset
import os

class SWDEDataset(BaseHTMLDataset):
    '''
    SWDE Dataset class for loading and accessing the SWDE dataset.
    Args:
        local_dir: Local directory where the SWDE repository is cloned.
        html_dir_in_repo: Subdirectory in the repo containing HTML files.
        data_dir_in_repo: Subdirectory in the repo containing Parquet data files.
        domain: Domain to load. Choices: 'auto','university','camera','book','job','nbaplayer','movie','restaurant'
        indices: Optional list of indices to restrict the dataset to a subset.
    '''
    
    
    def __init__(self,config: SWDEConfig):
        super().__init__(config=config)

        self._domain = config.domain
        self.data_source_path = config.data_source_path
        self.html_source_path = config.html_source_path

        # Load all subsets (domains) and combine into a single DatasetDict
        try:
            data_files = {}
            for file in os.listdir(self.data_source_path):
                if file.endswith(".parquet"):
                    domain_name = file.split("-")[0]   # e.g. "restaurant"
                    data_files[domain_name] = os.path.join(self.data_source_path, file)

            # Load all parquet files as separate splits
            self.dataset = load_dataset("parquet", data_files=data_files)
            print("Successfully loaded Parquet data into a DatasetDict.")
            print(f"Available subsets: {list(self.dataset.keys())}")
        except Exception as e:
            print(f"Could not load dataset from {self.data_source_path}. Error: {e}")
            raise  
        
    
    def _get_total_length(self) -> int:
        """Return number of samples"""
        return len(self.dataset[self._domain])

    def get_domains(self) -> List[str]:
        """Return the list of available domains in the dataset."""
        return list(self.dataset.keys())

    def set_domain(self, domain: str):
        """Set the domain for the dataset."""
        if domain not in ['auto','university','camera','book','job','nbaplayer','movie','restaurant']:
            raise ValueError(f"Domain '{domain}' not found in dataset. Available domains: {list(self.dataset.keys())}")
        self._domain = domain

    def get_domain(self) -> str:
        """Get the current domain of the dataset."""
        return self._domain


    def _get_item(self, idx: int) -> Sample:
        """Return (html, query, ground_truth) tuple for index"""
        if idx < 0 or idx >= self._get_total_length():
            raise IndexError("Index out of bounds")
        
        row = self.dataset[self._domain][idx]
        html_path = os.path.join(self.html_source_path, self._domain)
        folders = [f for f in os.listdir(html_path) if f.startswith(f"{self._domain}-{row['website_id'].split('_')[0]}")] 
        file_path = os.path.join(html_path, folders[0], f"{row['website_id'].split('_')[1]}.htm")

        if not os.path.isfile(file_path):
            print(f"HTML file not found for id {row['website_id']} at path: {file_path}")
            return Sample(**{
                "id": row['website_id'],
                "content": None,
                "query": row['schema'],
                "ground_truth": row['gt']
            })
        
        with open(file_path, 'r', encoding='utf-8') as file:
            html = file.read()

        # Access the HTML content based on the domain and id
        return Sample(**{
            "id": row['website_id'],
            "content": html,
            "query": row['schema'],
            "ground_truth": row['gt']
        })

    def get_website_name(self, idx: int) -> str:
        """Return the website name for a given index."""
        if idx < 0 or idx >= self._get_total_length():
            raise IndexError("Index out of bounds")
        
        row = self.dataset[self._domain][idx]
        return row['website_name']
    
