import os
import sys
import json
import yaml
import polars as pl
import multiprocessing
from stamina import retry
from typing import Dict, Any
from dotenv import load_dotenv
from ..base_pipeline import BasePipeline
from json_repair import json_repair
from ...experiment import Experiment
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Langchain
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from html_eval.pipelines.direct_extraction.preprocessor import Preprocessor

class DirectExtraction(BasePipeline):
    def __init__(self, preprocessor: Preprocessor, llm: BaseChatModel, cpu_workers:int=multiprocessing.cpu_count(), **kwargs):
        super().__init__(**kwargs)
        self.cpu_workers = multiprocessing.cpu_count()
        self.preprocessr:Preprocessor = preprocessor
        self.llm:BaseChatModel = llm

        with open(r"/home/khaled/projects/Eval/src/eval/methods/direct_extraction/prompts.yaml", 'r', encoding="utf-8") as file:
            self._yaml_prompt = yaml.safe_load(file)
        
        # FIXME: Change this bad name
        self.extraction_prompt_template = ChatPromptTemplate.from_template(self._yaml_prompt["gpt-extraction-extraction_prompt_template"])
    
    def set_experiment(self, experiment:Experiment):
        self.experiment=experiment

    def _extract_info(self, page: Document, schema: str):
        # defining Chain
        extract_info_chain = (
            self.extraction_prompt_template
            | self.llm
            | StrOutputParser()
            | json_repair.repair_json
            | json.loads
        )

        # Defining a Retrying function
        @retry(on=Exception, attempts=5, wait_initial=5, wait_max=300)
        def extract_retry(input_):
            return extract_info_chain.invoke(input_)

        # Running the chain with retry
        final_dict = {}
        # FIXME: Fix prompts to include page instead of chunk
        input_ = {"page": page, "schema": schema}
        try:
            final_dict = extract_retry(input_)
        except Exception as e:
            print(f"Failed to extract info after retries: {e}")

        return final_dict

    def extract(self, batch: pl.DataFrame) -> Dict[str, Any]:
        """
        Extract information from a batch of content.
        """
                
        parameters = [
            (Document(row[0]), row[1])
            for row in batch.select(["html", "query"]).iter_rows()
        ]
        results = [None] * len(parameters)
        with ProcessPoolExecutor(max_workers=min(self.cpu_workers, len(parameters))) as executor:
            # results = list(executor.map(self._preprocess_single, contents))
            jobs = {executor.submit(self._extract_info, args): idx for idx, args in enumerate(parameters)}
            tmp = {}
            for job in as_completed(jobs):
                idx = jobs[job]
                try:
                    tmp[idx] = job.result()
                except Exception as e:
                    tmp[idx] = f"[Process ERROR] {e}"
                    print(f"error:{e}")
            for i in range(len(parameters)):
                results[i] = tmp[i]
        pass