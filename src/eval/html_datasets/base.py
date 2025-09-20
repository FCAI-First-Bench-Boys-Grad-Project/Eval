from abc import ABC, abstractmethod
from typing import Iterator, List, Optional
from eval.configs.dataset_config import BaseDatasetConfig

class Sample(dict):
    """A single sample from the dataset."""
    sample_id: str
    html_content: str
    query: str
    ground_truth: str

    def __getitem__(self, key):
        # tuple-style indexing
        if isinstance(key, int):  
            return list(self.values())[key]
        return super().__getitem__(key)



class BaseHTMLDataset(ABC):
    def __init__(self, config: BaseDatasetConfig):
        """
        Base class for HTML datasets with optional subset support.
        
        Args:
            indices: Optional list of indices to restrict the dataset to a subset.
        """
        self._config: BaseDatasetConfig = config
        self._indices: Optional[List[int]] = config.indices

    @abstractmethod
    def _get_total_length(self) -> int:
        """Return total number of samples in the dataset (before subsetting)."""
        pass

    @abstractmethod
    def _get_item(self, idx: int) -> Sample:
        """Return (html, query, ground_truth) for raw index (ignores subset)."""
        pass

    def __len__(self) -> int:
        """Return number of samples (considering subset if defined)."""
        return len(self._indices) if self._indices is not None else self._get_total_length()

    def __getitem__(self, idx: int) -> Sample:
        """Return item, respecting subset if defined."""
        if idx < 0 or idx >= self._get_total_length():
            raise IndexError("Index out of bounds")
        real_idx = self._indices[idx] if self._indices is not None else idx
        return self._get_item(real_idx)

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over dataset (respecting subset)."""
        for idx in range(len(self)):
            yield self[idx]

    def batch_iterator(
        self, batch_size: int, shuffle: bool = False
    ) -> Iterator[List[Sample]]:
        """Iterate over batches (respecting subset)."""
        import random

        indices = self.get_all_indices()
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            yield [self[j] for j in indices[i:i+batch_size]]

    def subset(self, indices: List[int]) -> "BaseHTMLDataset":
        """Return a new dataset object restricted to the given indices."""
        cls = self.__class__
        new_obj = cls.__new__(cls)  # create instance without calling __init__
        new_obj.__dict__.update(self.__dict__)  # copy state
        new_obj._indices = indices
        return new_obj
    
    def get_all_indices(self) -> List[int]:
        """Return all valid indices in the dataset."""
        return self._indices if self._indices is not None else list(range(self._get_total_length()))
