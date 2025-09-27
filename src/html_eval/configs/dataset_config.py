from dataclasses import dataclass, field, asdict
import os
from typing import Optional, List, Dict, Any


@dataclass
class BaseDatasetConfig:
    indices: Optional[List[int]] = None
    batch_size: int = 1
    shuffle: bool = False
    evaluation_metrics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SWDEConfig(BaseDatasetConfig):
    local_dir: str = "hf_websrc"
    html_dir_in_repo: str = "files"
    data_dir_in_repo: str = "data"
    domain: str = "auto"
    dataset_name: str = field(init=False)
    html_source_path: str = field(init=False)
    data_source_path: str = field(init=False)

    def __post_init__(self):
        self.dataset_name = "swde"
        self.evaluation_metrics = ["page_level_f1"]
        self.html_source_path = os.path.join(self.local_dir, self.html_dir_in_repo)
        self.data_source_path = os.path.join(self.local_dir, self.data_dir_in_repo)

        if not os.path.isdir(self.html_source_path):
            raise FileNotFoundError(f"HTML directory not found in repo: {self.html_source_path}")
        if not os.path.isdir(self.data_source_path):
            raise FileNotFoundError(f"Data directory not found in repo: {self.data_source_path}")
        if not [f for f in os.listdir(self.html_source_path) if not f.endswith(".7z")]:
            raise ValueError(
                f"No HTML folder files found in the directory: {self.html_source_path}. "
                f"Please ensure to unzip the files."
            )

    def create_dataset(self):
        from html_eval.html_datasets.swde import SWDEDataset
        return SWDEDataset(self)


@dataclass
class WebSrcConfig(BaseDatasetConfig):
    html_source_path: str = "data/websrc/html_content.jsonl"
    data_source_path: str = "data/websrc/data.jsonl"
    dataset_name: str = field(init=False)

    def __post_init__(self):
        self.dataset_name = "websrc"
        self.evaluation_metrics = ["token_f1"]

        if not self.html_source_path.endswith(".jsonl"):
            raise ValueError(f"HTML source file must be a .jsonl file: {self.html_source_path}")
        if not self.data_source_path.endswith(".jsonl"):
            raise ValueError(f"Data source file must be a .jsonl file: {self.data_source_path}")

        if not os.path.isfile(self.html_source_path):
            raise FileNotFoundError(f"HTML source file not found: {self.html_source_path}")
        if not os.path.isfile(self.data_source_path):
            raise FileNotFoundError(f"Data source file not found: {self.data_source_path}")

    def create_dataset(self):
        from html_eval.html_datasets.websrc import WebSrcDataset
        return WebSrcDataset(self)
