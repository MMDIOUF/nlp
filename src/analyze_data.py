"""Lecture et contrôle du corpus CoNLL MasakhaNER en wolof.

Une phrase CoNLL est un bloc de lignes ``token label``. Une ligne vide sépare
deux phrases. Le schéma BIO indique le début (B) ou la continuation (I) d'une
entité. Ce module analyse les données sans les modifier automatiquement.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


VALID_LABELS = {
    "O",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-DATE",
    "I-DATE",
}


@dataclass(frozen=True)
class Sentence:
    """Une phrase alignant exactement un token avec une étiquette."""

    tokens: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class BioError:
    """Une transition BIO impossible localisée dans le fichier source."""

    sentence_index: int
    token_index: int
    token: str
    previous_label: str
    current_label: str


def load_conll(path: Path) -> list[Sentence]:
    """Charge un fichier CoNLL et refuse toute ligne ou étiquette invalide."""

    sentences: list[Sentence] = []
    tokens: list[str] = []
    labels: list[str] = []

    # utf-8-sig accepte aussi un éventuel marqueur BOM au début du fichier.
    with path.open(encoding="utf-8-sig") as corpus:
        for line_number, raw_line in enumerate(corpus, start=1):
            line = raw_line.strip()

            if not line:
                if tokens:
                    sentences.append(Sentence(tuple(tokens), tuple(labels)))
                    tokens, labels = [], []
                continue

            columns = line.rsplit(maxsplit=1)
            if len(columns) != 2:
                raise ValueError(f"{path}:{line_number}: ligne CoNLL invalide")

            token, label = columns
            if label not in VALID_LABELS:
                raise ValueError(
                    f"{path}:{line_number}: étiquette inconnue {label!r}"
                )
            tokens.append(token)
            labels.append(label)

    # Le dernier bloc peut ne pas être suivi d'une ligne vide.
    if tokens:
        sentences.append(Sentence(tuple(tokens), tuple(labels)))

    return sentences


def find_bio_errors(sentences: list[Sentence]) -> list[BioError]:
    """Repère les ``I-TYPE`` qui ne continuent pas une entité du même type."""

    errors: list[BioError] = []
    for sentence_index, sentence in enumerate(sentences, start=1):
        previous_label = "O"
        for token_index, (token, label) in enumerate(
            zip(sentence.tokens, sentence.labels), start=1
        ):
            if label.startswith("I-"):
                entity_type = label[2:]
                allowed_previous = {f"B-{entity_type}", f"I-{entity_type}"}
                if previous_label not in allowed_previous:
                    errors.append(
                        BioError(
                            sentence_index,
                            token_index,
                            token,
                            previous_label,
                            label,
                        )
                    )
            previous_label = label
    return errors


def count_entities(sentences: list[Sentence]) -> Counter[str]:
    """Compte les entités : chaque étiquette ``B-TYPE`` commence une entité."""

    return Counter(
        label[2:]
        for sentence in sentences
        for label in sentence.labels
        if label.startswith("B-")
    )


def print_report(name: str, sentences: list[Sentence]) -> None:
    """Affiche les statistiques utiles à la compréhension du corpus."""

    label_counts = Counter(label for sent in sentences for label in sent.labels)
    entity_counts = count_entities(sentences)
    bio_errors = find_bio_errors(sentences)
    token_count = sum(len(sentence.tokens) for sentence in sentences)
    o_ratio = label_counts["O"] / token_count if token_count else 0.0

    print(f"\n=== {name.upper()} ===")
    print(f"Phrases       : {len(sentences)}")
    print(f"Tokens        : {token_count}")
    print(f"Part de O     : {o_ratio:.2%}")
    print(f"Labels        : {dict(sorted(label_counts.items()))}")
    print(f"Entités       : {dict(sorted(entity_counts.items()))}")
    print(f"Erreurs BIO   : {len(bio_errors)}")

    for error in bio_errors[:5]:
        print(
            "  - phrase {0.sentence_index}, token {0.token_index} ({0.token!r}) : "
            "{0.previous_label} -> {0.current_label}".format(error)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse le corpus NER wolof")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="dossier contenant train.txt, dev.txt et test.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in ("train", "dev", "test"):
        path = args.data_dir / f"{split}.txt"
        print_report(split, load_conll(path))


if __name__ == "__main__":
    main()
