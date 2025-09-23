import requests
from bs4 import BeautifulSoup, Comment
import re
from htmlrag import clean_html as clean_html_rag
from html_chunking import get_html_chunks
from typing import List

DEFAULT_REMOVE_TAGS = ("script", "style")

def fetch_content( url: str,
                   timeout = 15) -> str:
    """
    Fetch the raw HTML content of a web page.

    Args:
        url (str): The URL of the web page to fetch.
        timeout (int, optional): The maximum time (in seconds) to wait for a response. 
                                Defaults to 15.

    Returns:
        str: The HTML content of the page if the request succeeds.
            If an error occurs, a string prefixed with "[FETCH_ERROR]" and the error message.
    """
    
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"[FETCH_ERROR] {e}"
        


def clean_html( html_content: str,
                extra_remove_tags = ["header" ,"footer"],
                strip_attrs: bool = True,
                strip_links: bool = True,
                keep_tags: bool = True,
                use_clean_rag: bool = True) -> str:
    """
    Clean raw HTML content by removing unwanted tags, attributes, comments, and optionally links.

    Args:
        html_content (str): The raw HTML content to clean.
        extra_remove_tags (List[str], optional): Additional tags to remove besides the defaults
                                                 (`script`, `style`). Defaults to ["header", "footer"].
        strip_attrs (bool, optional): If True, remove all tag attributes (e.g., class, id). 
                                      Defaults to True.
        strip_links (bool, optional): If True, replace <a> tags with their inner text. Defaults to True.
        keep_tags (bool, optional): If True, return cleaned HTML (with tags preserved). 
                                    If False, return plain text only. Defaults to True.
        use_clean_rag (bool, optional): If True, apply `htmlrag.clean_html` for additional 
                                        normalization. Defaults to True.

    Returns:
        str: The cleaned HTML or plain text, depending on `keep_tags`.
    """
    soup = BeautifulSoup(html_content or "", "html.parser")

    remove_tags = set(DEFAULT_REMOVE_TAGS) | set(extra_remove_tags)
    for tag_name in remove_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    if strip_attrs:
        for tag in soup.find_all(True):
            tag.attrs = {}

    if strip_links:
        for a in soup.find_all('a'):
            a.replace_with(a.get_text())

    for tag in soup.find_all(True):
        if not tag.get_text(strip=True):
            tag.decompose()

    if keep_tags:
        html_str = str(soup)
        html_str = re.sub(r'(?m)^[ \t]*\n', '', html_str)
        return html_str.strip()

    text = soup.get_text(separator='\n', strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    clean_text = '\n'.join(lines)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    if use_clean_rag:
        clean_text = clean_html_rag(clean_text)

    return clean_text.strip()

def chunk_html_content( html_content: str,
                        max_tokens: int = 500,
                        is_clean: bool = True,
                        attr_cutoff_len: int = 5) -> List[str]:
    """
    Split HTML content into smaller chunks suitable for processing (e.g., with LLMs).

    Args:
        html_content (str): The HTML content to chunk.
        max_tokens (int, optional): Maximum token length per chunk. Defaults to 500.
        is_clean (bool, optional): Whether the input HTML is already cleaned. Defaults to True.
        attr_cutoff_len (int, optional): Maximum length of attributes to retain. Defaults to 5.

    Returns:
        List[str]: A list of HTML/text chunks.
    """
    if not html_content:
        return []
    return get_html_chunks(html=html_content, max_tokens=max_tokens, is_clean_html=is_clean, attr_cutoff_len=attr_cutoff_len)