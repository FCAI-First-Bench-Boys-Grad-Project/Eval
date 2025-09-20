from dataclasses import dataclass

from typing import Optional

@dataclass
class BaseDatasetConfig:
    dataset_name: str
    indices: Optional[list[int]] = None
    batch_size: int = 1

class SWDEConfig(BaseDatasetConfig):
    dataset_name: str = "swde"
    local_dir: str = "hf_websrc"
    html_dir_in_repo: str = "files"
    data_dir_in_repo: str = "data"
    domain: str = "auto" 

class WebSrcConfig(BaseDatasetConfig):
    dataset_name: str = "websrc"
    html_source_path: str = "data/websrc/html_content.jsonl"
    data_source_path: str = "data/websrc/data.jsonl"



