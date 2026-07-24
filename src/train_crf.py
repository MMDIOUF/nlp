"""Construction et entraînement du modèle CRF pour le NER wolof."""

from __future__ import annotations

from pathlib import Path

from sklearn_crfsuite import CRF

from src.analyze_data import load_conll
from src.dataset import prepare_split


def build_crf(c1: float = 0.1, c2: float = 0.1) -> CRF:
    """Construit un CRF vierge avec nos hyperparamètres de départ."""

    return CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=100,
        all_possible_transitions=True,
    )


def train_crf(
    X_train: list[list[dict[str, object]]],
    y_train: list[list[str]],
    c1: float = 0.1,
    c2: float = 0.1,
) -> CRF:
    """Entraîne un nouveau CRF uniquement avec l'ensemble train."""

    if not X_train or not y_train:
        raise ValueError("X_train et y_train ne doivent pas être vides")
    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train doivent avoir le même nombre de phrases")

    model = build_crf(c1=c1, c2=c2)
    model.fit(X_train, y_train)
    return model


def predictions_are_aligned(
    predictions: list[list[str]],
    references: list[list[str]],
) -> bool:
    """Vérifie qu'une prédiction existe pour chaque token de référence."""

    return len(predictions) == len(references) and all(
        len(prediction) == len(reference)
        for prediction, reference in zip(predictions, references)
    )


def main() -> None:
    """Entraîne sur train et contrôle la forme des prédictions sur dev."""

    data_dir = Path("data/raw")
    train_sentences = load_conll(data_dir / "train.txt")
    dev_sentences = load_conll(data_dir / "dev.txt")

    X_train, y_train = prepare_split(train_sentences)
    X_dev, y_dev = prepare_split(dev_sentences)

    print(f"Entraînement sur {len(X_train)} phrases...")
    model = train_crf(X_train, y_train)
    dev_predictions = model.predict(X_dev)

    print(f"Prédictions produites pour {len(dev_predictions)} phrases de dev")
    print(f"Alignement des prédictions : {predictions_are_aligned(dev_predictions, y_dev)}")


if __name__ == "__main__":
    main()
