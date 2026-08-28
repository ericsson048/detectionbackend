import os
import requests

class AIHubMixClient:
    """Client pour l'API AIHubMix (compatible OpenAI /v1/chat/completions)."""

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("AIHUBMIX_API_KEY")
        self.base_url = (base_url or os.getenv("AIHUBMIX_BASE_URL",
                                               "https://aihubmix.com/v1")).rstrip("/")
        self.model = model or os.getenv("AIHUBMIX_MODEL", "coding-glm-5.3-free")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.4, max_tokens: int = 400) -> str:
        if not self.configured:
            raise RuntimeError("AIHUBMIX_API_KEY non configurée")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Réponse AIHubMix inattendue : {data}") from e
