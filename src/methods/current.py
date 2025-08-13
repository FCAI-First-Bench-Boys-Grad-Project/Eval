from src.methods.base import BasePipeline
from typing import Dict, Any
from src.methods.web2json.ai_extractor import *
from src.methods.web2json.preprocessor import *
from src.methods.web2json.postprocessor import *
from src.methods.web2json.pipeline import *
import os

from pydantic import BaseModel
class RerankerPipeline(BasePipeline):
    def __init__(self):
        self.prompt_template = """Extract the following information from the provided content.
                
        Content to analyze:
        {content}

        Instructions:
        - Extract only information that is explicitly present in the content
        - Preserve the original formatting and context where relevant
        - Return the extracted data in a structured format
        """

        try:
            self.preprocessor = BasicPreprocessor(config={'keep_tags': True}) 
            self.llm = NvidiaLLMClient(config={'api_key': os.getenv('NVIDIA_API_KEY'),'model_name': 'google/gemma-3n-e2b-it'})
            self.reranker = ModalRerankerClient("http://localhost:8000")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM client: {str(e)}")
        
        self.ai_extractor = None
        self.postprocessor = PostProcessor()
        self.pipeline = None

    def extract(self, html: str, query: str) -> Dict[str, Any]:
        """
        Extract information from content (converted to HTML), optionally using query.
        """

        if self.ai_extractor is None or self.pipeline is None:
            self.ai_extractor = LLMClassifierExtractor(
                reranker=self.reranker,
                llm_client=self.llm,
                prompt_template=self.prompt_template + f"Answer the question {query} based on the provided content.",
                classifier_prompt=f"Answer the question {query} based on the provided content.",
            )
            self.pipeline = Pipeline(self.preprocessor, self.ai_extractor, self.postprocessor)

        try:
            # make a BaseModel class that just has answer as a field
            class AnswerModel(BaseModel):
                answer: str
            result = self.pipeline.run(content = html,is_url=False, schema=AnswerModel)
            print("-"*80)
            print(f"Processed result: {result}")
            return result
        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}

    def _convert_to_html(self, content: str) -> str:
        """
        Converts input content to HTML string.
        """
        if content.strip().startswith("<html"):
            return content
        else:
            return f"<html><body>{content}</body></html>"
