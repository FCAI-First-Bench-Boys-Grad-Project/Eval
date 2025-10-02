import os
from dotenv import load_dotenv

load_dotenv()
# move to the directory where src is located
os.chdir('/home/abdo/PAPER/Eval/src')

# Setting an API key for NVIDIA 
os.environ['NVIDIA_API_KEY'] = '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'

from html_eval import Experiment
from html_eval.configs.experiment_config import ExperimentConfig
from html_eval.configs.dataset_config import SWDEConfig , WebSrcConfig
from html_eval.configs.pipeline_config import RerankerPipelineConfig , RerankerPreprocessorConfig , RerankerExtractorConfig , RerankerPostprocessorConfig
from html_eval.configs.llm_client_config import LLMClientConfig

TOTAL_GPU_UTIL = 0.9
GEN_GPU_UTIL = 0.2

import multiprocessing as mp
    
# The goal is to answer the following question (Does the reranker provide any kind of help?)
# Things we need to iterate on
# SWDE verticals / WEDBSRC
# Disable Method (chunking/reranker)
######################################## CONSTANTS
SWDE_DOMAINS = {
    "auto": 17900,
    "university": 16700,
    "camera": 5260,
    "book": 20000,
    "job": 20000,
    "nbaplayer": 4410,
    "movie": 20000,
    "restaurant": 20000
}
SWDE_SAMPLES = 500
WEBSRC_TOTAL = 50000
WEBSRC_SAMPLES = 5000
DISABLE_METHOD_VALUE = [False,True]


########################################## CONFIG

dataset_configs = []
for dom , val in SWDE_DOMAINS.items:
    dataset_configs.append(SWDEConfig(
        local_dir="/home/abdo/PAPER/Eval/data/swde/hf_SWDE",
        indices=list(range(0,val,val/SWDE_SAMPLES)),  # Use a subset of the dataset for
        domain=dom,
        batch_size=5
    ))


dataset_configs.append(WebSrcConfig(
    html_source_path='/home/abdo/PAPER/Eval/data/websrc/hf_websrc_dev/dev/dev_html_content.jsonl',
    data_source_path='/home/abdo/PAPER/Eval/data/websrc/hf_websrc_dev/dev/dev_dataset.jsonl',
    indices= list(range(0,WEBSRC_TOTAL,WEBSRC_TOTAL/WEBSRC_SAMPLES)),
    batch_size=5
))
print(f"Created {len(dataset_configs)} of dataset configs.")

reranker_preprocessor_config = RerankerPreprocessorConfig(
    fetch_workers=mp.cpu_count(),
    cpu_workers=mp.cpu_count()
)
llm_client_config = LLMClientConfig(
    model_name='google/gemma-3n-e4b-it',
)
reranker_extractor_config = RerankerExtractorConfig(
    llm_config=llm_client_config,
    generation_prompt_template="""
# Information Extraction System Prompt

## Core Instructions

You are an information extraction assistant. You must ONLY respond with a single VALID JSON object.

**Critical Requirements:**
- No markdown code blocks
- No explanations or extra text
- No preamble or commentary
- Only the raw JSON object

**Validation:**
- Ensure JSON is well-formed and parsable
- If a field cannot be extracted, set it to `null`
- For arrays/objects, use empty `[]` or `{{}}` if specified by schema

---

## Extraction Logic

### Case 1: Schema-Based Extraction (Query is a JSON Schema)

When the query contains a JSON Schema with `properties` field:

1. Extract information for each field defined in the schema
2. Follow the data types specified (`string`, `number`, `boolean`, etc.)
3. Include surrounding context if it provides a more complete answer
4. Match field descriptions to content semantically

**Output Format:** Follow the schema structure exactly

### Case 2: Natural Language Query (No Schema)

When the query is a question or instruction without a formal schema:

1. Identify the most relevant information that answers the query
2. Return the answer in this format:
   ```json
   {{
     "answer": "extracted information here"
   }}
   ```

---

## Context-Aware Interpretation

**Important:** Queries may use different English dialects. Interpret words based on context:

- **"fair"** → "reasonably priced" (price context), "just/equitable" (justice context), "beautiful" (aesthetic context)
- **"proper"** → "correct" (accuracy context), "very" (emphasis, e.g., British English)
- **"cheap"** → "inexpensive" (neutral) vs "low quality" (negative)

Always prioritize **content context** over assumptions.

---

## Content Extraction Rules

### Rule 1: Preserve Context
If the extracted answer is a subset of a larger node and the surrounding content provides clarity, include relevant surrounding text.

**Example:**
- Raw span: `"Stephen Hawking"`
- Better extraction: `"Author: Stephen Hawking"` (if "Author:" provides useful context)

### Rule 2: Handle HTML Gracefully
- Extract text content from HTML tags
- Ignore purely structural elements
- Preserve meaningful formatting context (labels, headers, etc.)

### Rule 3: Be Precise Yet Complete
- Don't add information not in the content
- Don't omit information that directly answers the query
- Balance brevity with completeness

---

## Examples

### Example 1: Schema-Based Extraction

**Query:**
```json
{{
  "id": "book.schema",
  "description": "Extract details about a book.",
  "type": "object",
  "properties": {{
    "title":{{
      "type": "string",
      "description": "The title of the book."
    }},
    "author": {{
      "type": "string",
      "description": "The author of the book."
    }}
  }}
}}
```

**Content:**
```
["<html><body><div><h1>A Brief History of Time</h1><span>Author: Stephen Hawking</span></div></body></html>"]
```

**Output:**
```json
{{
  "title": "A Brief History of Time",
  "author": "Author: Stephen Hawking"
}}
```

---

### Example 2: Schema with Date Extraction

**Query:**
```json
{{
  "id": "event.schema",
  "description": "Extract details about an event.",
  "type": "object",
  "properties": {{
    "event_name": {{
      "type": "string",
      "description": "The name of the event."
    }},
    "date": {{
      "type": "string",
      "description": "The date of the event."
    }}
  }}
}}
```

**Content:**
```
["<html><body><p>Event: Annual Tech Conference</p><span>Date: 2024-10-26</span></body></html>"]
```

**Output:**
```json
{{
  "event_name": "Event: Annual Tech Conference",
  "date": "Date: 2024-10-26"
}}
```

---

### Example 3: Natural Language Query (Dialect-Aware)

**Query:**
```
How fair was the book?
```

**Content:**
```
["<p>The book was reasonably priced at $9.99 and had a lovely story.</p>"]
```

**Output:**
```json
{{
  "answer": "$9.99"
}}
```

*Note: "fair" interpreted as "price" based on context.*

---

### Example 4: Boolean Answer Query

**Query:**
```
Did the review mention if the event was successful?
```

**Content:**
```
["<p>The conference was a huge success, with over 500 attendees.</p>"]
```

**Output:**
```json
{{
  "answer": "Yes"
}}
```

---

### Example 5: Missing Information

**Query:**
```json
{{
  "type": "object",
  "properties": {{
    "price": {{"type": "number"}},
    "isbn": {{"type": "string"}}
  }}
}}
```

**Content:**
```
["<p>This book costs $15.99</p>"]
```

**Output:**
```json
{{
  "price": 15.99,
  "isbn": null
}}
```

---


NOW EXTRACT THIS
```
Query:
{query}

Content:
{content}
```

**Remember:** Respond with ONLY the JSON object. No additional text, explanations, or markdown formatting.
THE TEXT YOU EXTRACT MUST EXACTLY BE IN THE CONTENT.
LEARN FROM THE EXAMPLES AND FOLLOW THE INSTRUCTIONS OR YOU WILL FACE YOUR DEATH.
""",
    classification_prompt_template= (
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
        ),
    reranker_quantization=None,
    # reranker_gpu_memory_utilization=TOTAL_GPU_UTIL - GEN_GPU_UTIL,
)



for data_cfg in dataset_configs:
    for method_status in DISABLE_METHOD_VALUE:
        name = "swde_" if isinstance(data_cfg,SWDEConfig) else "websrc_"
        name += data_cfg.domain if isinstance(data_cfg,SWDEConfig) else ""
        name += str(method_status)

        reranker_postprocessor_config = RerankerPostprocessorConfig(
            exact_extraction= isinstance(data_cfg,SWDEConfig)
        )
        reranker_pipeline_config = RerankerPipelineConfig(
            preprocessor_config=reranker_preprocessor_config,
            extractor_config=reranker_extractor_config,
            postprocessor_config=reranker_postprocessor_config,
            disable_method=method_status
            )


        exp_config = ExperimentConfig(
            experiment_name=name,
            seed=42,
            pipeline_config=reranker_pipeline_config,
            dataset_config=data_cfg,
            output_dir=name
        )
        exp = Experiment(exp_config,resume=False)
        exp.run()

########################################## RUN
