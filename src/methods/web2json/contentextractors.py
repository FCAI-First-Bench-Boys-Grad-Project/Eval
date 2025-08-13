import os
import re
import json
import pdfkit
import requests
import warnings
import tempfile
# import textract
import html2text
import inscriptis
import trafilatura
from pathlib import Path
from markdownify import markdownify
from json_repair import repair_json
from bs4 import BeautifulSoup, Comment
from html_chunking import get_html_chunks
from urllib.error import URLError, HTTPError
from html_to_markdown import convert_to_markdown
from readabilipy import simple_json_from_html_string
from docling.document_converter import DocumentConverter
from dateparser_scripts.update_supported_languages_and_locales import to_string


def clean_html(html_content: str) -> str:
    """
    Cleans up the given HTML content by:
      - Removing <script> and <style> tags and their content.
      - Removing HTML comments.
      - Extracting and returning the visible text with normalized whitespace.
    
    Args:
        html_content (str): The HTML content to clean.
    
    Returns:
        str: The cleaned, visible text from the HTML.
    """
    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    # Remove unwanted tags
    for tag in soup(["script", "style", "img", "a", "table", "tr", "td", "th", "thead", "tbody",
                     "tfoot", "header", "footer", "link", "rel"]):
        tag.decompose()

    # Remove elements that do not contain any visible text
    for element in soup.find_all():
        # If the element has no text (after stripping whitespace), remove it
        if not element.get_text(strip=True):
            element.decompose()
    
    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Extract text and normalize whitespace
    # text = soup.get_text(separator=" ", strip=True)
    # clean_text = re.sub(r'\s+', ' ', text)
    
    # return clean_text
    return str(soup)


def print_content_extractors():
    print(
        [
            "Default: the plain text of the HTML page",
            "Inscriptis",
            "Trafilatura",
        ]
    )


class ContentExtractor:
    def get_text(self, html):
        return clean_html(html)

    # TODO: Clean this mess
    def url_to_html(self, url,clean=False):
        # Define custom headers to mimic a browser request
        headers = {
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
            "Upgrade-Insecure-Requests": "1"
        }

        try:
            # Create a Request object with custom headers
            response = requests.get(url, headers=headers, timeout=10)

            html = None

            if response.status_code == 200:
                html = response.text
            else:
                print(f"Failed to retrieve HTML. Status code: {response.status_code}")
                return None

            if clean:    
                return self.get_text(html)
            
            return html

        except HTTPError as e:
            print(f"HTTP Error: {e.code} - {e.reason}")
            return None
        except URLError as e:
            print(f"URL Error: {e.reason}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None


class Inscriptis(ContentExtractor):
    def __init__(self):
        super()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Brave/119.0.0.0",
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        }

        warnings.warn("\nBeware, put only clean links with no trackers, or it may produce unexpected results.")

    def get_text(self, html):
        """Extract text from HTML using inscriptis."""
        return inscriptis.get_text(html)

    def url_to_html(self, url):
        response = requests.get(url, headers=self.headers)
        return response.text


class Docling(ContentExtractor):
    def __init__(self):
        super().__init__()

    # TODO: This is an unexpected behaviour but due to docling docs website being down, it's what works for now
    def get_text(self, text_content):
        result = None
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.html', delete=False, encoding='utf-8') as tmpfile:
            tmpfile.write(text_content)
            tmpfile.flush()
            tmpfile_path = tmpfile.name.replace("\\", "/")
            tmpfile_path = Path(tmpfile_path)
        try:
            converter = DocumentConverter()
            document = converter.convert(tmpfile_path).document
            tables = []
            for table_ix, table in enumerate(document.tables):
                table_text = table.export_to_markdown()
                tables.append(table_text)

            result = document.export_to_markdown()
            for table in tables:
                result += "\n\n" + table
        finally:
            os.remove(tmpfile_path)
        return result


class ReadabiliPy(ContentExtractor):
    def __init__(self):
        super().__init__()

    def get_text(self, html):
        content = simple_json_from_html_string(html, use_readability=True)
        json_object = json.dumps(content, indent=4)
        repaired = repair_json(json_object)
        return repaired


class Trafilatura(ContentExtractor):
    def __init__(self):
        super().__init__()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }

        warnings.warn("\nTrafilatura Content Extractor: Beware, put only clean links with no trackers, or it may produce unexpected results.")

        from copy import deepcopy
        from trafilatura.settings import DEFAULT_CONFIG
        config = deepcopy(DEFAULT_CONFIG)
        # config['DEFAULT']['MIN_EXTRACTED_SIZE'] = '5000' # Configurable but this value worked well for me
        self.config = config

    def url_to_html(self, url):
        response = requests.get(url, headers=self.headers)
        return response.text

    def get_text(self, html, output_format="markdown", min_extracted_size_char=20_000):
        # self.config['DEFAULT']['MIN_EXTRACTED_SIZE'] = f"{min_extracted_size_char}"
        # self.config['DEFAULT']['MIN_OUTPUT_SIZE'] = f"{min_extracted_size_char}"
        return trafilatura.extract(filecontent=html, favor_recall=True, config=self.config, output_format=output_format)
    

class Markdownify(ContentExtractor):
    def get_text(self, html):
        alt = re.sub(r"\n{3,}", "\n\n", html)
        md = markdownify(alt, strip=['href', 'table', 'tr', 'td', 'header', 'footer'])

        md = re.sub(r'!?\[[^\]]*\]\([^)]*\)', '', md)
        # Remove extra newlines
        md = re.sub(r"\n{3,}", "\n\n", md)
        md = md.strip()

        return md


class HTML2Text(ContentExtractor):
    def get_text(self, html):
        converter = html2text.HTML2Text()
        converter.ignore_tables=True
        converter.ignore_links=True
        converter.ignore_images=True
        converter.ignore_mailto_links=True
        return converter.handle(html)
    

class HTML_TO_Markdown(ContentExtractor):
    def get_text(self, html):
        alt = re.sub(r"\n{3,}", "\n\n", html)
        md = convert_to_markdown(alt, strip=['href', 'table', 'tr', 'td', 'header', 'footer'])
    
        md = re.sub(r'!?\[[^\]]*\]\([^)]*\)', '', md)
        # Remove extra newlines
        md = re.sub(r"\n{3,}", "\n\n", md)
        md = md.strip()

        return md


class PDFkitDocling(ContentExtractor):
    def get_text(self, html):
        soup = BeautifulSoup(html, "html.parser")

        # Remove <a>, <link>, <img>, and other unwanted tags
        for tag in soup.find_all(['a', 'link', 'img', 'base', 'meta', 'style', 'script', 'noscript', 'head']):
            tag.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()


        content = str(soup)

        # PDF path to save
        pdf_path = 'test.pdf'

        # Create PDF
        pdfkit.from_string(content, pdf_path)

        converter = DocumentConverter()

        return converter.convert(pdf_path).document.export_to_markdown()


class TrafilatraCHUNKS(ContentExtractor):
    def __init__(self):
        super().__init__()
        # self.trafi = Trafilatura()

    def get_text(self, html, max_tokens=1000):
        soup = BeautifulSoup(html, "html.parser")

        # Remove <a>, <link>, <img>, and other unwanted tags
        for tag in soup.find_all(['a', 'link', 'img', 'base', 'meta', 'style', 'script', 'noscript', 'head']):
            tag.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()


        content = str(soup)

        chunks = get_html_chunks(content, max_tokens=max_tokens, is_clean_html=True, attr_cutoff_len=50)

        cleaned = [trafilatura.extract(chunk) for chunk in chunks]
        cleaned = [chunk for chunk in cleaned if chunk is not None]
        

        combined_text = ""
        for chunk in cleaned:
            if chunk is None:
                continue
            combined_text += chunk + "\n"
        
        return combined_text


class TrafilaCHUNKSRobust(ContentExtractor):
    def __init__(self):
        super().__init__()
        # self.trafi = Trafilatura()

    def get_text(self, html, max_tokens=1000):
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.find_all(['style', 'script', 'head', 'img', 'base', 'noscript']):
            tag.decompose()

        for tag in soup.find_all(lambda tag: tag.attrs and any("nav" in str(v) for v in tag.attrs.values())):
            tag.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        content = str(soup)

        chunks = get_html_chunks(content, max_tokens=max_tokens, is_clean_html=True, attr_cutoff_len=50)

        cleaned = [trafilatura.extract(chunk) for chunk in chunks]
        cleaned = [chunk for chunk in cleaned if chunk is not None]
        
        combined_text = ""
        for chunk in cleaned:
            if chunk is None:
                continue
            combined_text += chunk + "\n"
        
        return combined_text

class TrafilaCHUNKSRobustV2(ContentExtractor):
    def __init__(self):
        super().__init__()
        # self.trafi = Trafilatura()

    def get_text(self, html, max_tokens=1000):
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.find_all(['style', 'script', 'head', 'img', 'base', 'noscript']):
            tag.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        content = str(soup)

        chunks = get_html_chunks(content, max_tokens=max_tokens, is_clean_html=True, attr_cutoff_len=50)

        cleaned = [trafilatura.extract(chunk) for chunk in chunks]
        cleaned = [chunk for chunk in cleaned if chunk is not None]
        
        combined_text = ""
        for chunk in cleaned:
            if chunk is None:
                continue
            combined_text += chunk + "\n"
        
        return combined_text

# Very Bad lol
# class Textract(ContentExtractor):
#     def get_text(self, html):
#         with tempfile.NamedTemporaryFile(mode='w+', suffix='.html', delete=False, encoding='utf-8') as tmpfile:
#             tmpfile.write(html)
#             tmpfile.flush()
#             tmpfile_path = tmpfile.name.replace("\\", "/")
#             tmpfile_path = Path(tmpfile_path)
#         try:
#             result = textract.process(tmpfile_path)
#         finally:
#             os.remove(tmpfile_path)
#         return result