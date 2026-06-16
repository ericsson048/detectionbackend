import requests

class GemmaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.4, max_tokens: int = 300) -> str:
        payload = {
            "model": "google/gemma-3-12b-it",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=15
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
