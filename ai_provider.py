"""Provider IA unifié avec bascule automatique entre Gemma et AIHubMix.

Priorité par défaut : Gemma (local) d'abord, puis AIHubMix si Gemma est
indisponible ou ne répond pas correctement.
"""
from gemma_client import GemmaClient
from aihubmix_client import AIHubMixClient


class AIProvider:
    def __init__(self, gemma_url: str, fallback_order=("gemma", "aihubmix")):
        self.gemma = GemmaClient(gemma_url)
        self.aihubmix = AIHubMixClient()
        self.fallback_order = fallback_order
        self._last_used = None

    @property
    def last_used(self) -> str:
        """Nom du fournisseur ayant servi de la dernière réponse."""
        return self._last_used

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.4, max_tokens: int = 400) -> str:
        """Génère une réponse en essayant les fournisseurs dans l'ordre.

        Retourne la première réponse valide. Lève une exception si tous
        les fournisseurs échouent.
        """
        errors = []
        for name in self.fallback_order:
            try:
                provider = self._get_provider(name)
                if provider is None:
                    raise RuntimeError(f"{name} non configuré")
                text = provider.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if text and text.strip():
                    self._last_used = name
                    return text
                raise RuntimeError(f"{name} a retourné une réponse vide")
            except Exception as e:
                errors.append(f"{name}: {e}")
                print(f"[AIProvider] Fallback {name} échoué : {e}")

        raise RuntimeError(" ; ".join(errors))

    def _get_provider(self, name: str):
        if name == "gemma":
            return self.gemma if self.gemma.base_url else None
        if name == "aihubmix":
            return self.aihubmix if self.aihubmix.configured else None
        return None
