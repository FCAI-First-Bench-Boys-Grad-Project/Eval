from eval.methods.base import BasePipeline
from eval.methods.web2json.ai_extractor import *
from eval.methods.web2json.preprocessor import *
from eval.methods.web2json.postprocessor import *
from eval.methods.web2json.llm import *
from pydantic import BaseModel
class RerankerPipeline(BasePipeline):
    
    def __init__(self):
        super().__init__()

        self.preprocessor = Preprocessor(chunk_size=500)
        self.extractor = RerankerExtractor(
            llm_client=NvidiaLLMClient(config={
            "model_name": "google/gemma-3n-e4b-it",
            }),
            prompt_template="""You are an assistant that must ONLY respond with a single VALID JSON object (no markdown, no explanation, no extra text).
            Validate that the JSON is well-formed. If a requested field cannot be extracted, set it to null (or an empty list/object if the schema specifies).

            Now extract information according below.

            Query:
            {query}

            Content:
            {content}

            If the query is a schema, extract the information according to the schema.
            If the query is a question, extract the answer to the question from the content and return it in the JSON object.
            The JSON object should be structured as follows:
            ```json
            {{
                "answer": "extracted answer",
            }}
            ```
            Return the JSON object now. DO NOT output anything else."""
        )
        self.postprocessor = PostProcessor()

    def extract(self, batch: pl.DataFrame) -> List[dict]:
        """
        Extract information from a batch of content.
        """
        # Preprocess the batch
        preprocessed_batch = self.preprocessor.process_batch(batch,content_col='html',return_polars=True)

        # Extract using AIExtractor
        extracted_data = self.extractor.extract(preprocessed_batch)
        # print(f"Extracted Data: {extracted_data}")
        # Post-process the extracted data
        postprocessed_data = self.postprocessor.process_dataframe(extracted_data)

        return postprocessed_data
        

    
