"""Module RAG (Retrieval-Augmented Generation) pour SkinDetect.

Récupère les extraits les plus pertinents de la base de connaissances
(knowledge_base.py) à partir de la prédiction / de la question de l'utilisateur,
puis les assemble en un "contexte" injecté dans le prompt du modèle.

Au-delà de la correspondance par mots-clés, on ajoute un enrichissement
conditionnel (conseils statistiques déjà validés) pour garantir des réponses
basées sur du contenu fiable.
"""
import re
from knowledge_base import KNOWLEDGE_CHUNKS

# Synonymes / formes canoniques par condition pour améliorer le rappel
_CANONICAL = {
    "chickenpox": ["chickenpox", "varicelle", "vésicule"],
    "cowpox": ["cowpox", "vaccine"],
    "hfmd": ["hfmd", "pieds-mains-bouche", "pieds mains bouche"],
    "healthy": ["healthy", "sain", "peau saine"],
    "measles": ["measles", "rougeole"],
    "monkeypox": ["monkeypox", "mpox", "variole du singe"],
}


def _normalize(text: str) -> str:
    """Normalise le texte : minuscules, sans accents, sans ponctuation."""
    text = text.lower()
    # retire les accents
    text = text.translate(str.maketrans(
        "àâäéèêëîïôöùûüçñ",
        "aaaeeeeiioouuucn"
    ))
    return re.sub(r"[^a-z0-9\s]", " ", text)


def _expand_query(query: str) -> list:
    """Étend la requête à ses synonymes canoniques."""
    terms = {query}
    q = _normalize(query)
    for canonical, aliases in _CANONICAL.items():
        for alias in aliases:
            if alias in q:
                terms.update(aliases)
                terms.add(canonical)
            if q in _normalize(alias):
                terms.add(canonical)
                terms.update(aliases)
    # ajout des mots significatifs de la question (mots-clés > 3 lettres)
    terms.update(
        w for w in _normalize(query).split()
        if len(w) > 3
    )
    return list(terms)


def retrieve(query: str, top_k: int = 3) -> list:
    """Retourne la liste des chunks les plus pertinents pour la requête.

    Le score est basé sur le nombre de correspondances entre les mots-clés
    du chunk et la requête étendue (et ses synonymes).
    """
    if not query or not query.strip():
        return []

    expanded = _expand_query(query)

    scored = []
    for chunk in KNOWLEDGE_CHUNKS:
        kw_norm = [_normalize(k) for k in chunk["keywords"]]
        score = sum(1 for k in kw_norm if k in expanded)
        # bonus si un mot-clé canonique de condition est présent
        for term in expanded:
            if term in kw_norm:
                score += 2
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def build_context(query: str, top_k: int = 3) -> str:
    """Assemble le contexte RAG formaté pour injection dans le prompt."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return ""

    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Document {i}] {c['content']}")
    return "\n\n".join(parts)
