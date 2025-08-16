import os
import time
import numpy as np
from google import genai
from openai import OpenAI
import time
import random
from openai import RateLimitError
from functools import wraps
from google.genai import types
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from html_chunking import get_html_chunks
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple, Optional
import re
import json
from langchain_text_splitters import HTMLHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
from tenacity import retry, wait_exponential, stop_after_attempt
import trafilatura


class LLMClient(ABC):
    """
    Abstract base class for calling LLM APIs.
    """
    def __init__(self, config: dict = None):
        """
        Initializes the LLMClient with a configuration dictionary.
        
        Args:
            config (dict): Configuration settings for the LLM client.
        """
        self.config = config or {}

    @abstractmethod
    def call_api(self, prompt: str) -> str:
        """
        Call the underlying LLM API with the given prompt.
        
        Args:
            prompt (str): The prompt or input text for the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

class RerankerClient(ABC):
    """
    Abstract base class for reranker APIs.
    """
    def __init__(self, config: dict = None):
        """
        Initializes the RerankerClient with a configuration dictionary.

        Args:
            config (dict): Configuration settings for the reranker client.
        """
        self.config = config or {}

    @abstractmethod
    def rerank(self, query: str, passages: List[str], top_k: int = 3) -> List[str]:
        """
        Rerank passages based on relevance to query.

        Args:
            query (str): Query string.
            passages (List[str]): List of passages.
            top_k (int): Number of top passages to return.

        Returns:
            List[str]: Top-k most relevant passages.
        """
        pass


class GeminiLLMClient(LLMClient):
    """
    Concrete implementation of LLMClient for the Gemini API.
    """

    def __init__(self, config: dict):
        """
        Initializes the GeminiLLMClient with an API key, model name, and optional generation settings.

        Args:
            config (dict): Configuration containing:
                - 'api_key': (optional) API key for Gemini (falls back to GEMINI_API_KEY env var)
                - 'model_name': (optional) the model to use (default 'gemini-2.0-flash')
                - 'generation_config': (optional) dict of GenerateContentConfig parameters
        """
        api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for Gemini must be provided in config['api_key'] or GEMINI_API_KEY env var."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = config.get("model_name", "gemini-2.0-flash")
        # allow custom generation settings, fallback to sensible defaults
        gen_conf = config.get("generation_config", {})
        self.generate_config = types.GenerateContentConfig(
            response_mime_type=gen_conf.get("response_mime_type", "text/plain"),
            temperature=gen_conf.get("temperature"),
            max_output_tokens=gen_conf.get("max_output_tokens"),
            top_p=gen_conf.get("top_p"),
            top_k=gen_conf.get("top_k"),
            # add any other fields you want to expose
        )

    def call_api(self, prompt: str) -> str:
        """
        Call the Gemini API with the given prompt (non-streaming).

        Args:
            prompt (str): The input text for the API.

        Returns:
            str: The generated text from the Gemini API.
        """
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        # Non-streaming call returns a full response object
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self.generate_config,
        )

        # Combine all output parts into a single string
        return response.text

def extract_markdown_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Find the first Markdown ```json ...``` block in `text`,
        parse it as JSON, and return the resulting dict.
        Returns None if no valid JSON block is found.
        """
        # 1) Look specifically for a ```json code fence
        fence_match = re.search(
            r"```json\s*(\{.*?\})\s*```",
            text,
            re.DOTALL | re.IGNORECASE
        )
        if not fence_match:
            return None

        json_str = fence_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

def retry_on_ratelimit(max_retries=5, base_delay=1.0, max_delay=10.0):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_retries - 1:
                        # give up
                        raise
                    # back off + jitter
                    sleep = min(max_delay, delay) + random.uniform(0, delay)
                    time.sleep(sleep)
                    delay *= 2
            # unreachable
        return wrapped
    return deco
class NvidiaLLMClient(LLMClient):
    """
    Concrete implementation of LLMClient for the NVIDIA API (non-streaming).
    """

    def __init__(self, config: dict):
        """
        Initializes the NvidiaLLMClient with an API key, model name, and optional generation settings.

        Args:
            config (dict): Configuration containing:
                - 'api_key': (optional) API key for NVIDIA (falls back to NVIDIA_API_KEY env var)
                - 'model_name': (optional) the model to use (default 'google/gemma-3-1b-it')
                - 'generation_config': (optional) dict of generation parameters like temperature, top_p, etc.
        """
        api_key = config.get("api_key") or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for NVIDIA must be provided in config['api_key'] or NVIDIA_API_KEY env var."
            )

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model_name = config.get("model_name", "google/gemma-3-1b-it")

        # Store generation settings with sensible defaults
        gen_conf = config.get("generation_config", {})
        self.temperature = gen_conf.get("temperature", 0)
        self.top_p = gen_conf.get("top_p", 0.7)
        self.max_tokens = gen_conf.get("max_tokens", 8192)

    def set_model(self, model_name: str):
        """
        Set the model name for the NVIDIA API client.

        Args:
            model_name (str): The name of the model to use.
        """
        self.model_name = model_name

    @retry_on_ratelimit(max_retries=20, base_delay=0.5, max_delay=5.0)
    def call_api(self, prompt: str) -> str:
        """
        Call the NVIDIA API with the given prompt (non-streaming).

        Args:
            prompt (str): The input text for the API.

        Returns:
            str: The generated text from the NVIDIA API.
        """
        print("prompt: ", prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"thinking":True}},
            # stream is omitted (defaults to False)
        )
        # print("DONE")
        # For the standard (non-streaming) response:
        # choices[0].message.content holds the generated text
        return response.choices[0].message.content
    
    def call_batch(self, prompts, max_workers=8):
        """
        Parallel batch with isolated errors: each prompt that still
        fails after retries will raise, but others succeed.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.call_api, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                    print("DONE")
                except RateLimitError:
                    # You could set results[idx] = None or a default string
                    results[idx] = f"<failed after retries>"
        return results
    

class NvidiaRerankerClient(RerankerClient):
    """
    Concrete implementation of LLMClient for the NVIDIA API (non-streaming).
    """

    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "nvidia/llama-3.2-nv-rerankqa-1b-v2")
        self.client = NVIDIARerank(
            model=self.model_name,
            api_key=os.getenv("NVIDIA_API_KEY"),
        )

    def set_model(self, model_name: str):
        """
        Set the model name for the NVIDIA API client.

        Args:
            model_name (str): The name of the model to use.
        """
        self.model_name = model_name

    @retry_on_ratelimit(max_retries=6, base_delay=0.5, max_delay=5.0)
    def rerank(self, query: str, passages: List[str], top_k: int = 3, threshold: float = 0.5) -> List[Document]:
        # 1. Prepare and send documents for scoring
        docs = [Document(page_content=p) for p in passages]
        scored_docs = self.client.compress_documents(
            query=str(query),
            documents=docs
        )

        # 2. Extract raw scores and compute sigmoid probabilities
        raw_scores = np.array([doc.metadata['relevance_score'] for doc in scored_docs], dtype=float)
        print(f"raw scores {raw_scores}")
        p_scores = 1 / (1 + np.exp(-raw_scores))
        print(f"Sigmoid scores: {p_scores}")

        # 3. Max normalization
        max_score = np.max(p_scores)
        if max_score == 0:
            norm_scores = np.zeros_like(p_scores)
        else:
            norm_scores = p_scores / max_score
        print(f"Normalized scores: {norm_scores}")

        # 4. Filter by threshold using normalized scores
        scored_pairs = [(doc, norm) for doc, norm in zip(scored_docs, norm_scores) if norm > threshold]
        print(f"Filtered pairs:\n{scored_pairs}")

        # 5. Return top_k documents (already sorted by model, no need to re-sort)
        top_docs = [doc.page_content for doc, _ in scored_pairs]
        return top_docs



    
    # TODO: will I need it ?
    # def call_batch(self, prompts, max_workers=8):
    #     pass

def retry_on_error(fn):
    """Simple retry decorator (exponential back-off, max 6 tries)."""
    return retry(
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        stop=stop_after_attempt(6),
        reraise=True,
    )(fn)


class ModalRerankerClient(RerankerClient):
    """Client for the Modal Qwen3-Reranker endpoint (non-streaming)."""

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url.rstrip("/")  # ensure no trailing slash

    def set_endpoint(self, url: str):
        self.endpoint_url = url.rstrip("/")

    @retry_on_error
    def rerank(
        self,
        query: str,
        passages: List[str],
        threshold: float = 0.5,
    ) -> List[Document]:
        """Call the remote endpoint and return filtered passages."""
        if not isinstance(query,str):
            query = str(query)
        payload = {"query": query, "passages": passages}
        print(payload)
        res = requests.post(self.endpoint_url + "/rerank", json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()

        # The endpoint already returns probabilities (0-1). Extract them.
        ranked = data.get("ranked_passages", [])
        # Extract scores
        scores = np.array([p["score"] for p in ranked], dtype=float)
        # Max normalization
        max_score = scores.max() if len(scores) > 0 else 1.0
        # max_score = 1
        if max_score == 0:
            norm_scores = np.zeros_like(scores)
        else:
            norm_scores = scores / max_score
        # Filter by threshold using normalized scores
        filtered = [
            (p, norm) for p, norm in zip(ranked, norm_scores) if norm >= threshold
        ]
        # Convert to LangChain Documents
        docs = [
            Document(page_content=p["passage"], metadata={"score": p["score"], "norm_score": norm})
            for p, norm in filtered
        ]
        
        # docs.reverse()

        return docs



    
    @retry_on_ratelimit(max_retries=6, base_delay=0.5, max_delay=5.0)
    def call_api(self, prompt: str) -> str:
        pass
    
    def call_batch(self, prompts, max_workers=8):
        pass


class AIExtractor:
    def __init__(self, llm_client: LLMClient, prompt_template: str):
        """
        Initializes the AIExtractor with a specific LLM client and configuration.

        Args:
            llm_client (LLMClient): An instance of a class that implements the LLMClient interface.
            prompt_template (str): The template to use for generating prompts for the LLM.
            should contain placeholders for dynamic content. 
            e.g., "Extract the following information: {content} based on schema: {schema}"
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template

    def extract(self, content: str, schema: BaseModel) -> str:
        """
        Extracts structured information from the given content based on the provided schema.

        Args:
            content (str): The raw content to extract information from.
            schema (BaseModel): A Pydantic model defining the structure of the expected output.

        Returns:
            str: The structured JSON object as a string.
        """
        prompt = self.prompt_template.format(content=content, schema=schema.model_json_schema())
        # print(f"Generated prompt: {prompt}")
        response = self.llm_client.call_api(prompt)
        return response
    
class LLMClassifierExtractor(AIExtractor):
    """
    Extractor that uses an LLM to classify and extract structured information from text content.
    This class is designed to handle classification tasks where the LLM generates structured output based on a provided schema.
    """
    def __init__(self, reranker: RerankerClient, llm_client: LLMClient, prompt_template: str, classifier_prompt: str, ):
        """
        Initializes the LLMClassifierExtractor with an LLM client and a prompt template.

        Args:
            llm_client (LLMClient): An instance of a class that implements the LLMClient interface.
            prompt_template (str): The template to use for generating prompts for the LLM.
        """
        super().__init__(llm_client, prompt_template)
        self.reranker = reranker
        self.classifier_prompt = classifier_prompt

    def chunk_content(self, content: str , max_tokens: int = 500, is_clean: bool = True) -> List[str]:
        """
        Splits the content into manageable chunks for processing.

        Args:
            content (str): The raw content to be chunked.

        Returns:
            List[str]: A list of text chunks.
        """
        # Use the get_html_chunks function to split the content into chunks
        return get_html_chunks(html=content, max_tokens=max_tokens, is_clean_html=is_clean, attr_cutoff_len=5)
    

    def classify_chunks(self, passages, top_k=3, hf: bool = False):  # reranker
        # print("TIME TO CLASSIFY")
        query = self.classifier_prompt

        if hf:
            # print("Using Hugging Face reranker for classification.")
            return self.reranker.rerank(query, passages, top_k=top_k)
        response = self.reranker.rerank(query,passages)
        print(f"response: {response}")
        # print("DONNNNE")
        # NVIDIA reranker path
        return response

    def extract(self, content, schema, hf: bool = False):
        """
        Extracts structured information from the given content based on the provided schema.

        Args:
            content (str): The raw content to extract information from.
            schema (BaseModel): A Pydantic model defining the structure of the expected output.
            hf (bool): Whether to use the Hugging Face reranker or NVIDIA (default).
        """
        # print("TIME TO EXTRACT")
        chunks = self.chunk_content(content, max_tokens=500)
        print(f"Content successfully chunked into {len(chunks)}.")
        # print(f"Content successfully chunked: {chunks}")
        # chunks = [trafilatura.extract(chunk,favor_recall=True) for chunk in chunks]
        # chunks = [chunk for chunk in chunks if chunk is not None]
        classified_chunks = self.classify_chunks(chunks, hf=hf)  # conditional reranker
        # extracting the content

        if isinstance(classified_chunks[0],Document):
            classified_chunks = [chunk.page_content for chunk in classified_chunks]
        print(f"Classified Chunks {len(classified_chunks)}")
        # print(classified_chunks)
        # print('='*80)
        # NOTE: More preprocesing
        # classified_chunks = [trafilatura.extract(chunk,favor_recall=True) for chunk in classified_chunks]
        # classified_chunks = [chunk for chunk in classified_chunks if chunk is not None]
        filtered_content = "\n\n".join(classified_chunks)

        if not filtered_content:
            print("Warning: No relevant chunks found. Returning empty response.")
            return "{}"

        prompt = self.prompt_template.format(content=filtered_content, schema=schema.model_json_schema())
        # print(f"Generated prompt for extraction: {prompt[:500]}...")
        llm_response = self.llm_client.call_api(prompt)
        # print(f"LLM response: {llm_response[:500]}...")
        
        return llm_response or "{}"

