
import polars as pl
import os
import math
import threading
from typing import Any, Dict, List, Optional
import torch
from html_eval.core.experiment import Experiment
from html_eval.configs.pipeline_config import RerankerExtractorConfig

            
class AIExtractor:

    def __init__(self, config: RerankerExtractorConfig):
        
        self.config = config

        self.llm_client = self.config.llm_config.create_llm_client()
        self.prompt_template = self.config.generation_prompt_template
        self.model_name: str = self.config.reranker_huggingface_model
        self.max_length = self.config.reranker_max_prompt_length
        self.default_top_k = self.config.reranker_default_top_k 
        
        self.vllm_kwargs = {
            "tensor_parallel_size": self.config.reranker_tensor_parallel_size if self.config.reranker_tensor_parallel_size is not None else torch.cuda.device_count(),
            "quantization": self.config.reranker_quantization,
            "gpu_memory_utilization": self.config.reranker_gpu_memory_utilization,
            "max_model_len": self.config.reranker_max_total_length,
            "enable_prefix_caching": self.config.reranker_enable_prefix_caching,
        }

        self.classification_prompt_template = self.config.classification_prompt_template
        self.reranker_classification_threshold = self.config.reranker_classification_threshold



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

    def set_experiment(self, experiment: Experiment ):
        self.experiment = experiment
        

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
        INST = self.classification_prompt_template

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
        # print(f"SHAPE BEFORE FILTER {batch.shape}")
        # print(batch)
        filtered_batch = batch.filter(pl.col('score_norm') >= threshold)
        # print(f"SHAPE AFTER FILTER {batch.shape}")
        # print(batch)
        return filtered_batch


    def _generate_output(self, batch: pl.DataFrame) -> pl.DataFrame:
        # Group by doc_id and concatenate chunks, while preserving the query column
        # print(f"BATCH SHAPE {batch.shape}")
        # print(f"Columns: {batch.columns}")
        df_grouped = batch.group_by("doc_id", maintain_order=True).agg(
            [pl.col(col).first() for col in batch.columns if col not in ("chunkcontent", "doc_id",'chunkid', 'score', 'score_norm')]
            + [pl.concat_str("chunkcontent", separator="\n").alias("full_content")]
        )
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
        # print("Final DataFrame with responses:")
        # print(df_response)
        return df_response

  
    
    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        
        # print(f"Shape before: format templates {df.shape} ")
        # Format the input Dataframe
        processed = []
        for row in df.iter_rows(named=True):
            for chunk in row['chunks']:
                processed += self._format_templates(row['query'], [chunk['chunkcontent']])
        # print(f"Shape before: classify {df.shape} ")
        # Score the passages
        scores = self._classify(processed)
        # print(f"Shape before: new DF {df.shape} ")


        # Step 2: explode the list into multiple rows
        expanded_df = df.explode("chunks")
        # print(f"Shape before: unnest {expanded_df.shape} ")

        # Step 3: unnest the struct inside the list
        expanded_df = expanded_df.unnest("chunks")
        # print(f"Shape before: float {expanded_df.shape} ")

        scores_df = expanded_df.with_columns(
            pl.Series('score', scores, dtype=pl.Float64)
        )

        # print(f"Shape before: norm {scores_df.shape} ")

        # Max normalization of scores for each docid TODO: need to add it do the config
        norm_df = scores_df.with_columns(
            (pl.col("score") / pl.col("score").max().over("doc_id")).alias("score_norm")
        )
        # print(f"Shape before: filter {norm_df.shape} ")

        # Filter the DataFrame based on the score threshold
        filtered_df = self._filter(norm_df, threshold=self.reranker_classification_threshold)
        # print(f"Shape before: generates {filtered_df.shape} ")

        generated_df = self._generate_output(filtered_df)
        # print(f"Shape before: extract exact {filtered_df.shape} ")

        final_df = generated_df

        return final_df 

        

 
        

