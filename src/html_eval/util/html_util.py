import requests
import re
from htmlrag import clean_html as clean_html_rag
from html_chunking import get_html_chunks
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from typing import List, Dict, Optional, Union, Tuple
from difflib import SequenceMatcher
import unicodedata
from rapidfuzz import fuzz

def normalize_html_text(text: str) -> str:
    """
    Normalize text by:
    - Converting weird Unicode whitespace (e.g. \u00a0, \u2009) to normal spaces
    - Collapsing multiple spaces into one
    - Stripping leading/trailing spaces
    Keeps capitalization and punctuation unchanged.
    """
    if not text:
        return ""
    
    # Normalize Unicode form
    normalized = text
    # Replace any Unicode whitespace with a plain space
    normalized = "".join(" " if ch.isspace() else ch for ch in normalized)
    
    # Collapse multiple spaces
    normalized = re.sub(r" +", " ", normalized)
    
    return normalized.strip()

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


def normalize_text(s: str) -> str:
    """
    Lowercase, replace punctuation with spaces, and collapse whitespace.
    This makes substring checks more robust (e.g. 6'10" -> '6 10').
    """
    if not s:
        return ""
    # replace non-word chars with spaces, lowercase, collapse spaces
    cleaned = re.sub(r'[^\w\s]', ' ', s).lower()
    return re.sub(r'\s+', ' ', cleaned).strip()

def find_closest_html_node(html_text, search_text):
    """
    Return the chunk (and its xpath/sub_index) that:
      - includes the normalized search_text as substring, and
      - has the highest fuzzy score among all such chunks.

    If no chunk includes search_text, returns {'text': search_text, 'xpath': None, 'sub_index': None, 'score': 0.0, 'found': False}.
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    norm_search = normalize_text(search_text)

    best_containing_score = 0.0
    best_containing_subset = 0
    best_containing_element = None
    best_containing_chunk = None
    best_containing_sub_index = None

    

    if not norm_search:
        # nothing meaningful to search for — return not-found payload
        return {'text': search_text, 'xpath': None, 'sub_index': None, 'score': 0.0, 'found': False}

    # Iterate only once: check containment and compute fuzzy score for those that contain
    for element in soup.find_all(True):
        for idx, chunk in enumerate(get_text_chunks(element)):
            if not chunk or not chunk.strip():
                continue
            intersection_tokens = len(set(norm_search.split()) & set(normalize_text(chunk).split()))
            if (norm_search in normalize_text(chunk)) or intersection_tokens:
                # candidate includes the search text -> compute fuzzy score
                score = SequenceMatcher(None, chunk.strip(), search_text.strip()).ratio()
                # print( score , intersection_tokens , chunk.strip())
                if score >= best_containing_score and intersection_tokens >= best_containing_subset:
                    best_containing_score = score
                    best_containing_subset = intersection_tokens
                    best_containing_element = element
                    best_containing_chunk = chunk.strip()
                    best_containing_sub_index = idx

    if best_containing_element is None:
        # nothing included the search text
        return {'text': search_text, 'xpath': None, 'sub_index': None, 'score': 0.0, 'found': False}

    return {
        'text': best_containing_chunk,
        'xpath': get_xpath(best_containing_element),
        'sub_index': best_containing_sub_index,
        'score': best_containing_score,
        'found': True
    }


def get_text_chunks(element):
    """
    Split by tags (include inner-tag text as separate chunks).
    """
    chunks = []
    buf = []

    for content in element.contents:
        if isinstance(content, NavigableString) and not isinstance(content, Comment):
            buf.append(str(content))
        elif isinstance(content, Tag):
            if buf:
                chunks.append(''.join(buf).strip())
                buf = []
            tag_text = content.get_text(separator=' ', strip=True)
            if tag_text:
                chunks.append(tag_text)
        else:
            if buf:
                chunks.append(''.join(buf).strip())
                buf = []

    if buf:
        chunks.append(''.join(buf).strip())

    return [c for c in chunks if c]


def get_xpath(element):
    components = []
    child = element if element.name else element.parent

    for parent in child.parents:
        siblings = parent.find_all(child.name, recursive=False)
        if len(siblings) == 1:
            components.append(child.name)
        else:
            index = siblings.index(child) + 1
            components.append(f'{child.name}[{index}]')
        child = parent

    components.reverse()
    if components and components[0] == '[document]':
        components.pop(0)

    return '/' + '/'.join(components)