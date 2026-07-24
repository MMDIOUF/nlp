"""Extraction et classement des erreurs NER au niveau des entités."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    """Une entité reconstruite depuis une séquence de labels BIO."""

    text: str
    entity_type: str
    start: int
    end: int


def extract_entities(tokens: list[str] | tuple[str, ...], labels: list[str]) -> list[Entity]:
    """Reconstruit les entités B-TYPE suivies de leurs I-TYPE."""

    entities: list[Entity] = []
    index = 0

    while index < len(labels):
        label = labels[index]
        if not label.startswith("B-"):
            index += 1
            continue

        entity_type = label[2:]
        start = index
        index += 1

        # Continuer seulement avec I- du même type.
        while index < len(labels) and labels[index] == f"I-{entity_type}":
            index += 1

        end = index - 1
        entities.append(
            Entity(
                text=" ".join(tokens[start : end + 1]),
                entity_type=entity_type,
                start=start,
                end=end,
            )
        )

    return entities


def spans_overlap(first: Entity, second: Entity) -> bool:
    """Indique si deux entités couvrent au moins un même token."""

    return first.start <= second.end and second.start <= first.end


def analyze_entity_errors(
    tokens: list[str] | tuple[str, ...],
    true_labels: list[str],
    predicted_labels: list[str],
) -> list[dict[str, str]]:
    """Classe les différences entre les entités attendues et prédites."""

    if len(tokens) != len(true_labels) or len(tokens) != len(predicted_labels):
        raise ValueError("tokens, true_labels et predicted_labels doivent être alignés")

    true_entities = extract_entities(tokens, true_labels)
    predicted_entities = extract_entities(tokens, predicted_labels)
    exact_true = {(entity.start, entity.end, entity.entity_type) for entity in true_entities}
    exact_predicted = {
        (entity.start, entity.end, entity.entity_type) for entity in predicted_entities
    }
    errors: list[dict[str, str]] = []

    for predicted in predicted_entities:
        predicted_key = (predicted.start, predicted.end, predicted.entity_type)
        if predicted_key in exact_true:
            continue

        same_limits = next(
            (
                expected
                for expected in true_entities
                if expected.start == predicted.start and expected.end == predicted.end
            ),
            None,
        )
        overlapping = next(
            (expected for expected in true_entities if spans_overlap(expected, predicted)),
            None,
        )

        if same_limits is not None:
            kind = "wrong_type"
            expected_text = f"{same_limits.text} · {same_limits.entity_type}"
        elif overlapping is not None:
            kind = "wrong_boundaries"
            expected_text = f"{overlapping.text} · {overlapping.entity_type}"
        else:
            kind = "invented"
            expected_text = "aucune entité"

        errors.append(
            {
                "kind": kind,
                "expected": expected_text,
                "predicted": f"{predicted.text} · {predicted.entity_type}",
            }
        )

    for expected in true_entities:
        expected_key = (expected.start, expected.end, expected.entity_type)
        if expected_key in exact_predicted:
            continue
        if any(spans_overlap(expected, predicted) for predicted in predicted_entities):
            continue

        errors.append(
            {
                "kind": "missed",
                "expected": f"{expected.text} · {expected.entity_type}",
                "predicted": "aucune entité",
            }
        )

    return errors
