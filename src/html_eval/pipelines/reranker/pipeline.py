from html_eval.pipelines.base_pipeline import BasePipeline
from html_eval.pipelines.reranker.preprocessor import BasePreprocessor
from html_eval.pipelines.reranker.ai_extractor import AIExtractor
from html_eval.pipelines.reranker.postprocessor import PostProcessor
from html_eval.configs.pipeline_config import RerankerPipelineConfig
from html_eval.core.types import Sample, SamplePrediction
from typing import List

class RerankerPipeline(BasePipeline):
    
    def __init__(self,config:RerankerPipelineConfig):
        super().__init__(config=config)

        self.preprocessor = BasePreprocessor(config=self.config.preprocessor_config)
        self.preprocessor.set_experiment(self.experiment)

        self.extractor = AIExtractor(config=self.config.extractor_config)
        self.extractor.set_experiment(self.experiment)

        self.postprocessor = PostProcessor(config=self.config.postprocessor_config)

    def extract(self, batch: List[Sample]) -> List[SamplePrediction]:
        """
        Extract information from a batch of content.
        """
        # Preprocess the batch
        # print(f"Batch PrePrecessing: {batch}")
        # print('='*80)
        preprocessed_batch = self.preprocessor.process(batch)
        # print(f"++++++++++EXPERIMENT Preprocessed Batch: {preprocessed_batch}")
        # print('='*80)
        # Extract using AIExtractor
        extracted_data = self.extractor.extract(preprocessed_batch)
        # print(f"Extracted Data: {extracted_data}")
        # print(f"Extracted Data Type: {extracted_data['response']}")
        # print('='*80)
        
        # Post-process the extracted data
        postprocessed_data = self.postprocessor.process_dataframe(extracted_data)
        # print(f"Postprocessed Data: {postprocessed_data}")
        # print(f"Postprocessed Data Type: {type(postprocessed_data)}")
        # print('='*80)
        return postprocessed_data
        

    
