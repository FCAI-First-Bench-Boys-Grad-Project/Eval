from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Any


class BaseHTMLDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Optional[str], Optional[str], Any]:
        """Return (html, query, ground_truth) tuple for index"""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[Optional[str], Optional[str], Any]]:
        """Iterate over (html, query, ground_truth) tuples"""
        pass

    @abstractmethod
    def batch_iterator(
        self, batch_size: int, shuffle: bool = False
    ) -> Iterator[List[Tuple[Optional[str], Optional[str], Any]]]:
        """Iterate over batches of (html, query, ground_truth) tuples"""
        pass