import re
import requests
from bs4 import BeautifulSoup , Comment
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from htmlrag import clean_html

class HTMLCleaner:
    DEFAULT_REMOVE_TAGS = [
        "script", "style"
    ]

    def __init__(self, config: dict = None):
        self.config = config or {}
        # allow custom tags to remove
        self.remove_tags = set(self.DEFAULT_REMOVE_TAGS) | set(self.config.get("extra_remove_tags", []))

    def _clean_html(self, html_content: str) -> str:
        """
        Cleans up the given HTML content by:
        - Removing specified tags and their content.
        - Stripping HTML comments.
        - Optionally stripping out all attributes.
        - Optionally flattening hyperlinks.
        - Removing empty tags.
        - Extracting and returning cleaned HTML or visible text.

        Args:
            html_content (str): The HTML content to clean.

        Returns:
            str: The cleaned HTML (if keep_tags=True) or normalized text.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted tags entirely
        for tag_name in self.remove_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Strip attributes if requested
        if self.config.get("strip_attrs", False):
            for tag in soup.find_all(True):
                tag.attrs = {}

        # Flatten hyperlinks if requested
        if self.config.get("strip_links", False):
            for a in soup.find_all('a'):
                a.replace_with(a.get_text())

        # Remove empty tags (no text and no non-empty children)
        for tag in soup.find_all(True):
            if not tag.get_text(strip=True):
                tag.decompose()

        # Convert soup to HTML string if preserving tags
        if self.config.get('keep_tags', False):
            html_str = str(soup)
            # Remove any empty lines
            html_str = re.sub(r'(?m)^[ \t]*\n', '', html_str)
            return html_str.strip()

        # Extract visible text
        text = soup.get_text(separator="\n", strip=True)
        # Remove empty lines
        lines = [line for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        # Normalize whitespace within lines
        clean_text = re.sub(r'\s+', ' ', clean_text)

        return clean_text.strip()

class Preprocessor(ABC):
    """
    Abstract base class for preprocessors.
    Defines the interface for transforming raw inputs into structured data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the preprocessor with optional configuration.

        Args:
            config: A dictionary of configuration settings.
            - keep_tags (bool): If True, keeps HTML tags in the output; otherwise, cleans them.
        """
        self.config = config if config is not None else {'keep_tags': False}

    def _fetch_content(self, url: str) -> str:
        """
        Fetches and parses the text content from a URL.

        Args:
            url: The URL to fetch content from.

        Returns:
            The clean, extracted text content from the page.

        Raises:
            ValueError: If the URL cannot be fetched or processed.
        """
        try:
            # Set a User-Agent header to mimic a browser, which can help avoid
            # being blocked by some websites.
            # Inside _fetch_content method
            headers =  headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.6",
                "Cache-Control": "max-age=0",
                "Sec-Ch-Ua": "\"Not(A:Brand\";v=\"99\", \"Brave\";v=\"133\", \"Chromium\";v=\"133\"",
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": "\"Windows\"",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            }
            
            # Make the HTTP GET request with a timeout.
            response = requests.get(url, headers=headers, timeout=15)
            
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            # Catch any network-related errors (DNS, connection, timeout, etc.)
            # and re-raise them as a more user-friendly ValueError.
            raise ValueError(f"Failed to fetch content from URL: {url}. Error: {e}")
        

    @abstractmethod
    def preprocess(self, content: str, is_url: bool) -> str:
        """
        Take raw content (HTML, text, etc.) and apply preprocessing steps.

        Args:
            content: The raw data to preprocess.

        Returns:
            A dictionary containing structured, cleaned data ready for downstream tasks.
        """
        pass

class BasicPreprocessor(Preprocessor):
    """
    Base preprocessor with common functionality.
    Can be extended for specific preprocessing tasks.
    """
    # TODO: Might need to think of how to improve this later
    def _clean_html(self, html_content: str) -> str:
        """
        Cleans up the given HTML content by:
        - Removing <script> and <style> tags and their content.
        - Removing HTML comments.
        - Extracting and returning the visible text with normalized whitespace if keep_tags is False.
        
        Args:
            html_content (str): The HTML content to clean.
        
        Returns:
            str: The cleaned, visible text from the HTML.
        """
        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Extract text and normalize whitespace
        if self.config.get('keep_tags', False):
            # If keep_tags is True, return the raw HTML
            return str(soup)
        
        text = soup.get_text(separator=" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', text)
        
        return clean_text

    def preprocess(self, content: str, is_url: bool) -> str:
        """
        Take raw content (HTML, text, etc.) and apply preprocessing steps.

        Args:
            content: The raw data to preprocess.

        Returns:
            A dictionary containing structured, cleaned data ready for downstream tasks.
        """
        
        html_content = content
        if is_url:
            # Fetch content from the URL
            html_content = self._fetch_content(content)


        # Clean the HTML content
        # cleaned_content = self._clean_html(html_content)
        cleaner = HTMLCleaner({
            'keep_tags': True if self.config.get('keep_tags', False) else False,
            'strip_attrs': True,
            'strip_links': True,
            'extra_remove_tags': ['header', 'footer']
        })
        clean = cleaner._clean_html(html_content=html_content)
        clean = clean_html(clean)
        # clean = clean_html(html_content)
        return clean.strip()  # Return the cleaned text content, stripped of leading/trailing whitespace

        

        