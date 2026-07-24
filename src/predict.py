"""Prédiction d'entités sur une nouvelle phrase wolof."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib

from src.error_analysis import extract_entities
from src.features import sentence_features


DEFAULT_MODEL_PATH = Path("models/crf_wolof.joblib")


def tokenize_text(text: str) -> list[str]:
    """Sépare les mots, nombres et ponctuations comme dans le corpus CoNLL."""

    return re.findall(r"[^\W_]+(?:[-'][^\W_]+)*|[^\w\s]", text, flags=re.UNICODE)


def predict_text(text: str, model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, object]:
    """Retourne les tokens, labels BIO et entités détectées."""

    tokens = tokenize_text(text.strip())
    if not tokens:
        raise ValueError("Le texte ne doit pas être vide")

    model = joblib.load(model_path)
    labels = list(model.predict_single(sentence_features(tokens)))
    entities = extract_entities(tokens, labels)

    return {
        "tokens": tokens,
        "labels": labels,
        "entities": [
            {"text": entity.text, "type": entity.entity_type}
            for entity in entities
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NER wolof avec le CRF entraîné")
    parser.add_argument("--text", required=True, help="phrase wolof à analyser")
    args = parser.parse_args()

    result = predict_text(args.text)
    print("Tokens et labels :", list(zip(result["tokens"], result["labels"])))
    print("Entités détectées :", result["entities"])


if __name__ == "__main__":
    main()
