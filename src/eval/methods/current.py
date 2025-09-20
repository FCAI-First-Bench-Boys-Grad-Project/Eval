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
            # "model_name": "qwen/qwen3-coder-480b-a35b-instruct",
            "model_name": "google/gemma-3n-e2b-it",
            }),
            prompt_template="""You are an assistant that must ONLY respond with a single VALID JSON object (no markdown, no explanation, no extra text).
            Validate that the JSON is well-formed. If a requested field cannot be extracted, set it to null (or an empty list/object if the schema specifies).

            Now extract information according below.

            Query:
            {query}

            Content:
            {content}

            If the query is a schema, extract the information according to the schema, meaning that for each field in the schema, you must extract the corresponding information from the content and fill it in the JSON object.
            IF THE EXTRACTED ANSWER IS A SUBSET OF A NODE AND THE REST OF THE NODE'S CONTENT WILL PROVIDE A BETTER ANSWER INCLUDE IT.
            The JSON object should be structured as follows:
            ```json
            {{
                "field1": "value1",
                "field2": "value2",
                ...
            }}
            ```

            IF THE QUERY IS NOT A SCHEMA, EXTRACT THE MOST RELEVANT INFORMATION FROM THE CONTENT THAT ANSWERS THE QUERY.
            AND RETURN IT IN A JSON OBJECT WITH A SINGLE KEY "answer".
            THE EXTRACTED ANSWER COULD BE A 'yes' or 'no'.
            The JSON object should be structured as follows:
            ```json
            {{
                "answer": "extracted answer",
            }}
            ```
            The answer should contain the extracted information as complete and accurate to the content as possible.
            Return the JSON object now. DO NOT output anything else.
            
            STICK TO THE ANSWER SCHEMA IF NO SCHEMA IS PROVIDED

            """
        )
        self.postprocessor = PostProcessor()

    def extract(self, batch: pl.DataFrame) -> List[dict]:
        """
        Extract information from a batch of content.
        """
        # Preprocess the batch
        print(f"Batch PrePrecessing: {batch}")
        print('='*80)
        preprocessed_batch = self.preprocessor.process_batch(batch,content_col='html',return_polars=True)
        print(f"Preprocessed Batch: {preprocessed_batch}")
        print('='*80)
        # Extract using AIExtractor
        extracted_data = self.extractor.extract(preprocessed_batch)
        print(f"Extracted Data: {extracted_data}")
        print(f"Extracted Data Type: {extracted_data['response']}")
        print('='*80)
        
        # Post-process the extracted data
        postprocessed_data = self.postprocessor.process_dataframe(extracted_data)
        print(f"Postprocessed Data: {postprocessed_data}")
        print(f"Postprocessed Data Type: {type(postprocessed_data)}")
        print('='*80)
        return postprocessed_data
        

    
