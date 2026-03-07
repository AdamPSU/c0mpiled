import os
import json
import httpx
from typing import Dict, Any

class AncestryTreeGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.system_prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "system_prompt.txt")
        
        with open(self.system_prompt_path, "r") as f:
            self.system_prompt = f.read().strip()

    async def generate_tree(self, query: str, papers_by_year: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls the OpenRouter API to generate a scientific ancestry tree.
        """
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")

        # Construct the user message
        user_message = {
            "query": query,
            "papers_by_year": papers_by_year
        }

        payload = {
            "model": "google/gemini-3-flash-preview", # Optimized model for reasoning and structured output (Gemini 3 Fast)
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(user_message)}
            ],
            "response_format": { "type": "json_object" }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ancestry.ai", # Site URL for OpenRouter ranking
            "X-Title": "Ancestry Paper Search"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()
                
                # Parse the content from the LLM response
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                if "error" in error_json:
                    error_detail = error_json["error"].get("message", error_detail)
            except:
                pass
            raise Exception(f"OpenRouter API error: {error_detail}")
        except Exception as e:
            raise Exception(f"Failed to generate ancestry tree: {str(e)}")
