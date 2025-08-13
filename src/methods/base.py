from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePipeline(ABC):
    def __init__(self, **kwargs):
        pass  # common setup if needed

    @abstractmethod
    def extract(self, html: str, query: str) -> Dict[str, Any]:
        """
        Extracts structured information from HTML content based on a query.

        This method should be implemented by subclasses to:
        - Preprocess the HTML input.
        - Extract relevant information according to the query.
        - Return the extracted data in a structured dictionary format.

        Parameters:
            html (str): The HTML content to extract information from.
            query (str): The query specifying what information to extract.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted information.
        """
        pass


