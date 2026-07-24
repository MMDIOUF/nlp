"""Baseline NER simple : mémoriser le label le plus fréquent de chaque mot."""

from collections import Counter, defaultdict


class MemorizationBaseline:
    """Attribue à chaque mot son label le plus fréquent observé dans train."""

    def __init__(self) -> None:
        self.word_to_label: dict[str, str] = {}

    def fit(self, sentences) -> "MemorizationBaseline":
        if not sentences:
            raise ValueError("Le jeu d'entraînement ne peut pas être vide.")

        counts = defaultdict(Counter)
        for sentence in sentences:
            for token, label in zip(sentence.tokens, sentence.labels):
                counts[token.lower()][label] += 1

        self.word_to_label = {
            word: label_counts.most_common(1)[0][0]
            for word, label_counts in counts.items()
        }
        return self

    def predict(self, sentences) -> list[list[str]]:
        if not self.word_to_label:
            raise ValueError("La baseline doit être entraînée avant predict().")

        return [
            [self.word_to_label.get(token.lower(), "O") for token in sentence.tokens]
            for sentence in sentences
        ]
