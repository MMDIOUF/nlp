"""Préparation des entrées X et des réponses y pour le modèle CRF."""

from __future__ import annotations

from src.analyze_data import Sentence
from src.features import sentence_features


def prepare_split(
    sentences: list[Sentence],
) -> tuple[list[list[dict[str, object]]], list[list[str]]]:
    """Transforme un ensemble de phrases en caractéristiques X et labels y."""

    X = [sentence_features(sentence.tokens) for sentence in sentences]
    y = [list(sentence.labels) for sentence in sentences]
    return X, y
