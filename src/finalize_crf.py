"""Entraînement, évaluation finale et sauvegarde du CRF sélectionné."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.analyze_data import load_conll
from src.dataset import prepare_split
from src.evaluate import evaluate_entities
from src.train_crf import train_crf


FINAL_CONFIG = {"c1": 0.05, "c2": 0.2}


def main() -> None:
    data_dir = Path("data/raw")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    train_sentences = load_conll(data_dir / "train.txt")
    test_sentences = load_conll(data_dir / "test.txt")
    X_train, y_train = prepare_split(train_sentences)
    X_test, y_test = prepare_split(test_sentences)

    # La configuration a été choisie auparavant avec dev puis figée.
    model = train_crf(X_train, y_train, **FINAL_CONFIG)
    test_predictions = [list(sequence) for sequence in model.predict(X_test)]
    test_metrics = evaluate_entities(y_test, test_predictions)

    model_path = output_dir / "crf_wolof.joblib"
    metadata_path = output_dir / "crf_wolof_metadata.json"
    joblib.dump(model, model_path)

    metadata = {
        "model": "CRF linéaire",
        "algorithm": "lbfgs",
        "c1": FINAL_CONFIG["c1"],
        "c2": FINAL_CONFIG["c2"],
        "max_iterations": 100,
        "all_possible_transitions": True,
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_report": test_metrics["report"],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Précision test : {test_metrics['precision']:.4f}")
    print(f"Rappel test    : {test_metrics['recall']:.4f}")
    print(f"F1 strict test : {test_metrics['f1']:.4f}")
    print(test_metrics["report"])
    print(f"Modèle sauvegardé : {model_path}")
    print(f"Métadonnées        : {metadata_path}")


if __name__ == "__main__":
    main()
