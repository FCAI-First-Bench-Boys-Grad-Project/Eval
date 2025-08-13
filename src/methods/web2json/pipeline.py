from src.methods.web2json.ai_extractor import *
from src.methods.web2json.postprocessor import *
from src.methods.web2json.preprocessor import *
from pydantic import BaseModel

class Pipeline:
    # constructor
    def __init__(self,
                 preprocessor: Preprocessor,
                 ai_extractor: AIExtractor,
                 postprocessor: PostProcessor):
        self.preprocessor = preprocessor
        self.ai_extractor = ai_extractor
        self.postprocessor = postprocessor

    def run(self, content: str, is_url: bool, schema:BaseModel, hf=False) -> dict:
        """
        Run the entire pipeline: preprocess, extract, and postprocess.

        Args:
            content (str): The raw content to process.
            is_url (bool): Whether the content is a URL or raw text.
            schema (BaseModel): The schema defining the structure of the expected output.

        Returns:
            dict: The final structured data after processing.
        """
        # Step 1: Preprocess the content
        preprocessed_content = self.preprocessor.preprocess(content, is_url)
        # print(f"Preprocessed content: {preprocessed_content}...")
        print('+'*80)
        # Step 2: Extract structured information using AI
        extracted_data = self.ai_extractor.extract(preprocessed_content, schema, hf=hf)
        # print(f"Extracted data: {extracted_data[:100]}...")
        print('+'*80)
        # Step 3: Post-process the extracted data
        final_output = self.postprocessor.process(extracted_data)
        print(f"Final output: {final_output}")
        print('+'*80)
        
        return final_output

        