from json_repair import repair_json
import json

class PostProcessor:

    def process(self, response: str) -> dict:
        json_response = {}
        try:
            # Extract the JSON from the generated text.  Handle variations in output format.
            json_string = response
            if "```json" in response:
                json_string = response.split("```json")[1].split("```")[0]
            elif "{" in response and "}" in response:
                # try to grab the json
                start_index = response.find("{")
                end_index = response.rfind("}") + 1
                json_string = response[start_index:end_index]

            json_response = json.loads(repair_json(json_string)) # Added for robustness
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(f"Generated text: {response}")
            json_response = {}

            
        return json_response

