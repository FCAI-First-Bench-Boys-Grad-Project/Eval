
from functools import wraps
import polars as pl
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from typing import List, Dict ,Any
from eval.methods.web2json.llm import LLMClient , NvidiaLLMClient
import polars as pl
import os
import math
import threading
from typing import Any, Dict, List, Optional
import polars as pl
import torch

class AIExtractor(ABC):
    def __init__(self, llm_client: Any, prompt_template: str):
        """
        Abstract base class for extractors that use an LLM to extract 
        structured information from a dataset.

        Args:
            llm_client (Any): An instance of a class that implements the LLMClient interface.
            prompt_template (str): Template for generating prompts for the LLM.
                                   Should contain placeholders for dynamic content.
                                   Example: 
                                   "Extract the following information: {content} based on schema: {schema}"
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template

    @abstractmethod
    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Abstract method to extract structured information from the given Polars DataFrame.

        Args:
            df (pl.DataFrame): The input Polars DataFrame containing raw content.

        Returns:
            pl.DataFrame: A Polars DataFrame containing structured information.
        """
        pass
    
class RerankerExtractor(AIExtractor, ABC):
    """
    Reranker that loads a vLLM-based reranker model in-process and uses it
    to score passages for each query. Returns a Polars DataFrame with columns:
      ["query", "passage", "score", "rank"]
    NOTE: Prompt template should has placeholders for "query" and "content".
    """

    def __init__(self, llm_client: Any, prompt_template: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_client, prompt_template)
        self.config = config.copy() if config else {}
        # default model (override via config["model_name"])
        self.model_name: str = self.config.get("model_name", "abdo-Mansour/Qwen3-Reranker-0.6B-HTML")
        # max token cap (same as server)
        self.max_length = self.config.get("max_length", 8192)
        # top_k returned per query (None => return all)
        self.default_top_k = self.config.get("top_k", None)
        # GPU / vllm options
        self.vllm_kwargs = self.config.get("vllm_kwargs", {
            "tensor_parallel_size": torch.cuda.device_count(),
            "quantization": "bitsandbytes",  # optional
            "gpu_memory_utilization": 0.7,
            "max_model_len": 2464,
            "enable_prefix_caching": True,
        })
        # placeholders to be set by _load_reranker
        self.tok = None
        self.llm = None
        self.suffix_ids = None
        self.yes_id = None
        self.no_id = None
        self.sampling = None
        self.llm_lock = threading.Lock()

        # load the reranker model/tokenizer into memory
        self._load_reranker()

    def _load_reranker(self):
        """
        Load tokenizer, vLLM LLM and sampling params into this instance.
        """
        # ensure environment flags similar to your script
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("VLLM_USE_V1", "1")

        # local imports (fail early if not installed)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        MODEL_NAME = self.model_name

        # Tokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        tok.padding_side = "left"
        tok.pad_token = tok.eos_token

        # LLM
        # pass through any kwargs set in self.vllm_kwargs
        llm = LLM(model=MODEL_NAME, **self.vllm_kwargs)

        # Suffix and yes/no token ids (keeps same behavior as your file)
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_ids = tok.encode(suffix, add_special_tokens=False)

        yes_id = tok("yes", add_special_tokens=False).input_ids[0]
        no_id = tok("no", add_special_tokens=False).input_ids[0]

        sampling = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[yes_id, no_id],
        )

        # assign to instance
        self.tok = tok
        self.llm = llm
        self.suffix_ids = suffix_ids
        self.yes_id = yes_id
        self.no_id = no_id
        self.sampling = sampling
        self.llm_lock = threading.Lock()

    def _format_templates(self, query: str, passages: List[str]) -> List[List[Dict[str, str]]]:
        """
        Build the chat-style templates for each passage (list-of-templates).
        Returns list of templates matching the vLLM chat API shape used in your server.
        """
        INST = (
            "You are a precision HTML content reranker. Your task is to evaluate HTML chunks "
            "for their potential to populate a given schema with meaningful data.\n\n"
            "## Core Objective:\n"
            "Score HTML content based on its likelihood to contain extractable information "
            "that matches the target schema requirements.\n\n"
            "## Instructions:\n"
            "1. Content Analysis: Examine the HTML chunk's text content, attributes, and semantic structure\n"
            "2. Schema Mapping: Assess how well the content aligns with schema field requirements\n"
            "3. Information Density: Evaluate the quantity and quality of extractable data\n"
            "4. Relevance Scoring: Assign a binary relevance score based on extraction potential\n"
        )

        def _format(q: str, d: str):
            return [
                {"role": "system", "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Answer only "yes" or "no".'},
                {"role": "user",   "content": f"<Instruct>: {INST}\n\n<Query>: {q}\n\n<Document>: {d}"},
            ]

        templates = [_format(query, p) for p in passages]
        return templates

    def _classify(self, processed_batch: List) -> List[float]:
        if not processed_batch:
            return []

        # ensure model loaded
        if self.llm is None or self.tok is None:
            raise RuntimeError("Reranker model/tokenizer not loaded")

        # Tokenize using tokenizer's chat helper
        # apply_chat_template returns token ids lists when tokenize=True
        tokenized = self.tok.apply_chat_template(processed_batch, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        # cap + append suffix ids
        tokenized = [ids[: self.max_length] + self.suffix_ids for ids in tokenized]

        # Prepare TokensPrompt objects
        from vllm.inputs.data import TokensPrompt
        msgs = [TokensPrompt(prompt_token_ids=ids) for ids in tokenized]

        # Call llm.generate (serialize with lock to be safe)
        def _call_generate():
            with self.llm_lock:
                return self.llm.generate(msgs, self.sampling, use_tqdm=False)

        outs = _call_generate()

        # Compute probabilities (softmax over yes/no logits) per passage
        scores: List[float] = []
        for o in outs:
            # defensive access to last token logprobs
            lp = o.outputs[0].logprobs[-1]
            true_logits = lp.get(self.yes_id, type("L", (), {"logprob": -10})).logprob
            false_logits = lp.get(self.no_id,  type("L", (), {"logprob": -10})).logprob

            # convert to probabilities (numerical stable enough for just two tokens)
            y = math.exp(true_logits)
            n = math.exp(false_logits)
            prob_yes = y / (y + n) if (y + n) != 0 else 0.0
            scores.append(prob_yes)
        
        return scores

    def _filter(self, batch: pl.DataFrame , threshold = 0.5) -> pl.DataFrame:
        """
        Filter the batch based on a threshold.
        Returns a DataFrame with only the rows that have a score above the threshold.
        """
        if 'score_norm' not in batch.columns:
            raise ValueError("Batch must contain 'score' column for filtering.")

        filtered_batch = batch.filter(pl.col('score') >= threshold)
        return filtered_batch


    def _generate_output(self, batch: pl.DataFrame) -> pl.DataFrame:
        # Group by doc_id and concatenate chunks, while preserving the query column
        df_grouped = batch.group_by("doc_id").agg([
            pl.concat_str("chunkcontent", separator="\n").alias("full_content"),
            pl.col("query").first().alias("query")  # Preserve the query column
        ])
    
        # Create the prompt using the prompt template
        df_prompt = df_grouped.with_columns(
            pl.struct(["query", "full_content"]).map_elements(
                lambda s: self.prompt_template.format(query=s["query"], content=s["full_content"]),
                return_dtype=pl.String  # Specify return type for clarity
            ).alias("prompt")
        )
        
        prompts = df_prompt["prompt"].to_list()
        # print("Generating prompts for LLM...")
        # Call the LLM client to get the responses
        responses = self.llm_client.call_batch(prompts)
        
        df_response = df_prompt.with_columns(
            pl.Series("response", responses, dtype=pl.Utf8)
        )
    
        return df_response
    
    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        
        # Format the input Dataframe
        processed = []
        for row in df.iter_rows(named=True):
            for chunk in row['content']['chunks']:
                processed += self._format_templates(row['query'], [chunk['chunkcontent']])
        
        # Score the passages
        scores = self._classify(processed)

        # Create a new DataFrame with the results
        expanded_df = df.with_columns(
            pl.col("content").struct.field("chunks").alias("chunks")
        )

        # Step 2: explode the list into multiple rows
        expanded_df = expanded_df.explode("chunks")

        # Step 3: unnest the struct inside the list
        expanded_df = expanded_df.unnest("chunks")

        scores_df = expanded_df.with_columns(
            pl.Series('score', scores, dtype=pl.Float64)
        )


        # Max normalization of scores for each docid TODO: need to add it do the config
        norm_df = scores_df.with_columns(
            (pl.col("score") / pl.col("score").max().over("doc_id")).alias("score_norm")
        )

        # Filter the DataFrame based on the score threshold
        filtered_df = self._filter(norm_df, threshold=0.5)

        return self._generate_output(filtered_df)

        

 
        

