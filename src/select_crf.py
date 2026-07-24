"""Comparaison reproductible de quelques configurations CRF sur dev."""

from __future__ import annotations

from sklearn_crfsuite import CRF

from src.evaluate import evaluate_entities
from src.train_crf import train_crf


CRF_CONFIGS = {
    "initial": {"c1": 0.1, "c2": 0.1},
    "l1_forte": {"c1": 0.5, "c2": 0.1},
    "l2_forte": {"c1": 0.1, "c2": 0.5},
    "equilibree": {"c1": 0.05, "c2": 0.2},
}


def select_best_crf(
    X_train: list[list[dict[str, object]]],
    y_train: list[list[str]],
    X_dev: list[list[dict[str, object]]],
    y_dev: list[list[str]],
) -> tuple[str, CRF, list[dict[str, float | str]]]:
    """Entraîne les configurations sur train et choisit le meilleur F1 dev."""

    results: list[dict[str, float | str]] = []
    best_name = ""
    best_model: CRF | None = None
    best_f1 = -1.0

    for name, config in CRF_CONFIGS.items():
        model = train_crf(X_train, y_train, **config)
        predictions = [list(sequence) for sequence in model.predict(X_dev)]
        metrics = evaluate_entities(y_dev, predictions)
        current_f1 = float(metrics["f1"])

        results.append(
            {
                "configuration": name,
                "c1": config["c1"],
                "c2": config["c2"],
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": current_f1,
            }
        )

        if current_f1 > best_f1:
            best_name = name
            best_model = model
            best_f1 = current_f1

    if best_model is None:
        raise RuntimeError("Aucune configuration CRF n'a été entraînée")

    return best_name, best_model, results
