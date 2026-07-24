"""Évaluation stricte des entités prédites avec le schéma IOB2."""

from __future__ import annotations

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2


def evaluate_entities(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict[str, float | str]:
    """Calcule les métriques au niveau des entités complètes."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true et y_pred doivent avoir le même nombre de phrases")

    options = {"mode": "strict", "scheme": IOB2, "zero_division": 0}
    return {
        "precision": precision_score(y_true, y_pred, **options),
        "recall": recall_score(y_true, y_pred, **options),
        "f1": f1_score(y_true, y_pred, **options),
        "report": classification_report(
            y_true,
            y_pred,
            digits=4,
            **options,
        ),
    }
