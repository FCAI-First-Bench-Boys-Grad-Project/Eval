
import polars as pl
from abc import ABC, abstractmethod
import os
import math
import threading
from typing import Any, Dict, List, Optional
import torch
from eval.experiment import Experiment
from lxml import html
import json
from json_repair import repair_json
import re
from bs4 import BeautifulSoup, Tag, NavigableString
from typing import Optional

def is_schema(text: str) -> bool:
    # Case 1: Python dict_keys
    if text.strip().startswith("dict_keys("):
        return True
    
    # Case 2: Looks like JSON or Python dict
    if (text.strip().startswith("{") and text.strip().endswith("}")) \
       or (text.strip().startswith("[") and text.strip().endswith("]")):
        try:
            json.loads(text)  # if it's valid JSON
            return True
        except Exception:
            return True  # maybe Python dict-like, not strict JSON
    
    # Case 3: Contains lots of colons/commas (schema-like pattern)
    if len(re.findall(r":", text)) >= 2:
        return True
    
    return False

def find_best_node_excluding_children(
    html_text: str,
    value: str,
    *,
    # Defaults changed: return plain text (no tags) by default
    return_html_without_children: bool = False,
    return_text_without_children: bool = True,
    fallback_to_full_node_if_no_direct_text: bool = True
) -> Optional[str]:
    """
    Find the node with best overlap to `value`, and return the node *without its child elements*.
    Default: return the node's direct text (no tags, no child text).
    - return_html_without_children=True -> returns HTML string of the node with child tags removed
      (keeps node attributes, preserves direct text nodes only).
    - return_text_without_children=True -> returns the direct text content only (no HTML).
    - If no direct text exists under the chosen node:
        - if fallback_to_full_node_if_no_direct_text True -> returns the full node text (descendant text).
        - otherwise returns an empty string (honors "exclude children" strictly).
    """
    if not html_text or not value:
        return None

    soup = BeautifulSoup(html_text, "html.parser")
    needle = value.strip().lower()

    def words(s: str):
        return re.findall(r"[A-Za-z0-9\-]+", s.lower())

    target_words = words(needle)
    if not target_words:
        return None

    def overlap_score(candidate: str, t_words: list[str]) -> float:
        c_set = set(words(candidate))
        t_set = set(t_words)
        if not c_set or not t_set:
            return 0.0
        return len(c_set & t_set) / len(t_set)

    candidates = []
    for tag in soup.find_all():
        # full_text: all descendant text joined (used for filtering & fallback)
        full_text = " ".join(tag.stripped_strings).strip()
        if needle not in full_text.lower():
            continue

        # gather direct text nodes only (recursive=False)
        direct_parts = [s for s in tag.find_all(string=True, recursive=False) if s and s.strip()]
        direct_text = " ".join(p.strip() for p in direct_parts).strip()

        candidate_for_scoring = direct_text if direct_text else full_text
        score = overlap_score(candidate_for_scoring, target_words)

        candidates.append({
            "score": score,
            "full_text": full_text,
            "direct_text": direct_text,
            "direct_parts": direct_parts,  # list of NavigableString
            "tag": tag
        })

    if not candidates:
        return None

    # choose best: highest score, then prefer smallest full_text (concise node)
    best = max(candidates, key=lambda c: (c["score"], -len(c["full_text"])))

    tag = best["tag"]
    direct_text = best["direct_text"]
    direct_parts = best["direct_parts"]
    full_text = best["full_text"]

    # If user wants HTML without children:
    if return_html_without_children:
        new_tag = soup.new_tag(tag.name)
        for k, v in tag.attrs.items():
            new_tag.attrs[k] = v

        if direct_parts:
            for part in direct_parts:
                new_tag.append(NavigableString(part))
            return str(new_tag)
        else:
            if fallback_to_full_node_if_no_direct_text:
                return full_text
            else:
                return str(new_tag)  # empty tag with attributes

    # If user wants plain text without children (default behavior):
    if return_text_without_children:
        if direct_text:
            return direct_text
        else:
            if fallback_to_full_node_if_no_direct_text:
                return full_text
            else:
                return ""  # strict exclude children

    # Fallback (shouldn't be hit because of defaults)
    return direct_text if direct_text else (full_text if fallback_to_full_node_if_no_direct_text else "")



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
        self.experiment = None  # Placeholder for experiment instance

    def set_experiment(self, experiment: Experiment) -> None:
        """
        Set the experiment instance for logging or other purposes.
        """
        self.experiment = experiment
        
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
            "max_model_len": 2048,
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
        print(f"SHAPE BEFORE FILTER {batch.shape}")
        print(batch)
        filtered_batch = batch.filter(pl.col('score_norm') >= threshold)
        print(f"SHAPE AFTER FILTER {batch.shape}")
        print(batch)
        return filtered_batch


    def _generate_output(self, batch: pl.DataFrame) -> pl.DataFrame:
        # Group by doc_id and concatenate chunks, while preserving the query column
        # print(f"BATCH SHAPE {batch.shape}")
        df_grouped = batch.group_by("doc_id",maintain_order=True).agg([
            pl.concat_str("chunkcontent", separator="\n").alias("full_content"),
            pl.col("query").first().alias("query")  # Preserve the query column
        ])
        # print(f"GROUPED BATCH SHAPE {df_grouped.shape}")
        # Create the prompt using the prompt template
        df_prompt = df_grouped.with_columns(
            pl.struct(["query", "full_content"]).map_elements(
                lambda s: self.prompt_template.format(query=s["query"], content=s["full_content"]),
                return_dtype=pl.String  # Specify return type for clarity
            ).alias("prompt")
        )
        # print(f"DF PROMPT SHAPE {df_prompt.shape}")
        prompts = df_prompt["prompt"].to_list()
        # print("Generating prompts for LLM...")
        # Call the LLM client to get the responses
        responses = self.llm_client.call_batch(prompts)
        
        df_response = df_prompt.with_columns(
            pl.Series("response", responses, dtype=pl.Utf8)
        )
        print("Final DataFrame with responses:")
        print(df_response)
        return df_response


    def _extract_exact_target(self, df: pl.DataFrame) -> pl.DataFrame:
        def process_row(full_content: list[str], response: str , query: str):
            try:
                if "```json" in response:
                    try:
                        response_json = response.split("```json", 1)[1].split("```", 1)[0]
                    except Exception:
                        response_json = response

                # fallback: try to capture the first {...} block
                if "{" in response_json and "}" in response_json:
                    start_index = response_json.find("{")
                    end_index = response_json.rfind("}") + 1
                    response_json = response_json[start_index:end_index]

                repaired = repair_json(response_json)
                response_json = json.loads(repaired)
            except Exception:
                raise ValueError(f"incorrect json format {response} , type of response ({type(response)})")
            print(f"Full Content: {full_content}")
            print(f"Preprocessed Response JSON: {response_json}")
            results = {}
            is_query_schema = is_schema(query)
            for key, value in response_json.items():
                if not value:
                    continue
                best_match = None
                best_score = -1
                for html_text in full_content:
                    candidate = find_best_node_excluding_children(html_text, str(value))
                    print(f"Key: {key}, Value: {value}, Candidate: {candidate}")
                    if candidate:
                        # compute score again (safe guard)
                        score = len(set(candidate.lower().split()) & set(str(value).lower().split())) / max(
                            1, len(set(str(value).lower().split()))
                        )
                        if score > best_score:
                            best_score = score
                            best_match = candidate
                if not is_query_schema:
                    key = 'answer'
                results[key] = best_match if best_match else value
            print(f"Extracted Targets: {results}")
            # turn it back to json string 
            results = json.dumps(results)
            return results
        print(f"Phase 0 {df}")
        print(f'Phase 1 {df.with_columns(pl.struct(["full_content", "response", "query"]))}')
        print("Phase2:", df.with_columns(
            pl.struct(["full_content", "response","query"])
            .map_elements(lambda row: process_row(row["full_content"], row["response"], row["query"]))
            .alias("response")
        ))

        return df.with_columns(
            pl.struct(["full_content", "response","query"])
            .map_elements(lambda row: process_row(row["full_content"], row["response"], row["query"]))
            .alias("response")
        )

  
    
    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        
        print(f"Shape before: format templates {df.shape} ")
        # Format the input Dataframe
        processed = []
        for row in df.iter_rows(named=True):
            for chunk in row['content']['chunks']:
                processed += self._format_templates(row['query'], [chunk['chunkcontent']])
        print(f"Shape before: classify {df.shape} ")
        # Score the passages
        scores = self._classify(processed)
        print(f"Shape before: new DF {df.shape} ")

        # Create a new DataFrame with the results
        expanded_df = df.with_columns(
            pl.col("content").struct.field("chunks").alias("chunks")
        )
        print(f"Shape before: explode {expanded_df.shape} ")

        # Step 2: explode the list into multiple rows
        expanded_df = expanded_df.explode("chunks")
        print(f"Shape before: unnest {expanded_df.shape} ")

        # Step 3: unnest the struct inside the list
        expanded_df = expanded_df.unnest("chunks")
        print(f"Shape before: float {expanded_df.shape} ")

        scores_df = expanded_df.with_columns(
            pl.Series('score', scores, dtype=pl.Float64)
        )

        print(f"Shape before: norm {scores_df.shape} ")

        # Max normalization of scores for each docid TODO: need to add it do the config
        norm_df = scores_df.with_columns(
            (pl.col("score") / pl.col("score").max().over("doc_id")).alias("score_norm")
        )
        print(f"Shape before: filter {norm_df.shape} ")

        # Filter the DataFrame based on the score threshold
        filtered_df = self._filter(norm_df, threshold=0.5) #FIXME: errorrr
        print(f"Shape before: generates {filtered_df.shape} ")

        generated_df = self._generate_output(filtered_df)
        print(f"Shape before: extract exact {filtered_df.shape} ")

        # final_df = self._extract_exact_target(generated_df)
        final_df = generated_df

        return final_df 

        

 
        

