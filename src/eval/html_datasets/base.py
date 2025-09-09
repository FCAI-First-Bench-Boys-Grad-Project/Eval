from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Any


class BaseHTMLDataset(ABC):
    def __init__(self, indices: Optional[List[int]] = None):
        """
        Base class for HTML datasets with optional subset support.
        
        Args:
            indices: Optional list of indices to restrict the dataset to a subset.
        """
        self._indices: Optional[List[int]] = indices

    @abstractmethod
    def _get_total_length(self) -> int:
        """Return total number of samples in the dataset (before subsetting)."""
        pass

    @abstractmethod
    def _get_item(self, idx: int) -> Tuple[Optional[str], Optional[str], Any]:
        """Return (html, query, ground_truth) for raw index (ignores subset)."""
        pass

    def __len__(self) -> int:
        """Return number of samples (considering subset if defined)."""
        return len(self._indices) if self._indices is not None else self._get_total_length()

    def __getitem__(self, idx: int) -> Tuple[Optional[str], Optional[str], Any]:
        """Return item, respecting subset if defined."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        real_idx = self._indices[idx] if self._indices is not None else idx
        return self._get_item(real_idx)

    def __iter__(self) -> Iterator[Tuple[Optional[str], Optional[str], Any]]:
        """Iterate over dataset (respecting subset)."""
        for idx in range(len(self)):
            yield self[idx]

    def batch_iterator(
        self, batch_size: int, shuffle: bool = False
    ) -> Iterator[List[Tuple[Optional[str], Optional[str], Any]]]:
        """Iterate over batches (respecting subset)."""
        import random

        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(self[idx])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def subset(self, indices: List[int]) -> "BaseHTMLDataset":
        """Return a new dataset object restricted to the given indices."""
        cls = self.__class__
        new_obj = cls.__new__(cls)  # create instance without calling __init__
        new_obj.__dict__.update(self.__dict__)  # copy state
        new_obj._indices = indices
        return new_obj
