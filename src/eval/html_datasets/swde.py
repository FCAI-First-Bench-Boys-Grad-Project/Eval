from eval.html_datasets.base import BaseHTMLDataset
from typing import Iterator, List, Optional, Tuple, Any
import polars as pl
import random
from datasets import load_dataset
from huggingface_hub import snapshot_download
import os
from datasets import DatasetDict

class SWDEDataset(BaseHTMLDataset):
    
    
    def __init__(self, 
                local_dir: str = "hf_websrc",
                html_dir_in_repo: str = "files",
                data_dir_in_repo: str = "data",
                domain: str = "auto",
                indices: Optional[List[int]] = None): 
        super().__init__(indices=indices)
    
        self.downloaded_repo_path = local_dir
        # 2. Define the paths to your HTML source and your Parquet data.
        self.html_source_path = os.path.join(self.downloaded_repo_path, html_dir_in_repo)
        self.data_source_path = os.path.join(self.downloaded_repo_path, data_dir_in_repo)
        # 3. Validate paths.
        if not os.path.isdir(self.html_source_path):
            raise FileNotFoundError(f"HTML directory not found in repo: {self.html_source_path}")
        if not os.path.isdir(self.data_source_path):
            raise FileNotFoundError(f"Data directory not found in repo: {self.data_source_path}")
        # Check if the HTML source directory contains files
        if not [f for f in os.listdir(os.path.join(self.downloaded_repo_path, html_dir_in_repo)) if not f.endswith(".7z")]:
            raise ValueError(f"No HTML folder files found in the directory: {self.html_source_path}. Please ensure to unzip the files.")

        # 4. Load the dataset directly from the local Parquet directory.
        # `load_dataset` will automatically find all subdirectories in `data_source_path`
        # and treat them as different subsets (e.g., train, test, or by domain).
        try:

            # Load all subsets (domains) and combine into a single DatasetDict
            # Map split names to files
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
        
        self._domain = domain
        self.evaluation_metrics = ['page_level_f1']
    
    def _get_total_length(self) -> int:
        """Return number of samples"""
        return len(self.dataset[self._domain])

    def set_domain(self, domain: str):
        """Set the domain for the dataset."""
        if domain not in ['auto','university','camera','book','job','nbaplayer','movie','restaurant']:
            raise ValueError(f"Domain '{domain}' not found in dataset. Available domains: {list(self.dataset.keys())}")
        self._domain = domain

    def get_domain(self) -> str:
        """Get the current domain of the dataset."""
        return self._domain


    def _get_item(self, idx: int) -> Tuple[Optional[str], Optional[str], Any]:
        """Return (html, query, ground_truth) tuple for index"""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        
        row = self.dataset[self._domain][idx]
        query = row['schema']
        ground_truth = row['gt']  # Adjust based on your evaluation needs

        
        html_path = os.path.join(self.html_source_path, self._domain)
        folders = [f for f in os.listdir(html_path) if f.startswith(f"{self._domain}-{row['website_id'].split('_')[0]}")] 
        file_path = os.path.join(html_path, folders[0], f"{row['website_id'].split('_')[1]}.htm")

        if not os.path.isfile(file_path):
            print(f"HTML file not found for id {row['website_id']} at path: {file_path}")
            return None, query, ground_truth
        
        with open(file_path, 'r', encoding='utf-8') as file:
            html = file.read()

        # Access the HTML content based on the domain and id
        return html, query, ground_truth

    
