"""Caractéristiques simples utilisées par le modèle CRF.

Un CRF ne comprend pas directement les mots. Pour chaque token, on construit
un petit dictionnaire d'indices : forme du mot, préfixe, suffixe et contexte
immédiat. Ces informations sont observables et faciles à expliquer.
"""

from __future__ import annotations


def word_shape(word: str) -> str:
    """Résume la forme d'un mot avec X (majuscule), x (minuscule) et d (chiffre).

    Exemple : ``Sénégal2026`` devient ``Xxxxxxxdddd``.
    Les autres caractères, comme le tiret, sont conservés.
    """

    shape = []
    for character in word:
        if character.isupper():
            shape.append("X")
        elif character.islower():
            shape.append("x")
        elif character.isdigit():
            shape.append("d")
        else:
            shape.append(character)
    return "".join(shape)


def token_features(tokens: list[str] | tuple[str, ...], index: int) -> dict[str, object]:
    """Construit les caractéristiques du token situé à ``index``.

    Le modèle observe le mot courant ainsi que les mots directement avant et
    après. Il ne voit jamais les vrais labels : cela évite toute fuite de la
    réponse attendue vers les données d'entrée.
    """

    word = tokens[index]
    lower_word = word.lower()

    features: dict[str, object] = {
        "bias": 1.0,
        "word.lower": lower_word,
        "word.prefix1": lower_word[:1],
        "word.prefix2": lower_word[:2],
        "word.suffix2": lower_word[-2:],
        "word.suffix3": lower_word[-3:],
        "word.shape": word_shape(word),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
    }

    if index == 0:
        features["BOS"] = True  # Beginning Of Sentence
    else:
        previous_word = tokens[index - 1]
        features.update(
            {
                "-1:word.lower": previous_word.lower(),
                "-1:word.istitle": previous_word.istitle(),
                "-1:word.isupper": previous_word.isupper(),
            }
        )

    if index == len(tokens) - 1:
        features["EOS"] = True  # End Of Sentence
    else:
        next_word = tokens[index + 1]
        features.update(
            {
                "+1:word.lower": next_word.lower(),
                "+1:word.istitle": next_word.istitle(),
                "+1:word.isupper": next_word.isupper(),
            }
        )

    return features


def sentence_features(tokens: list[str] | tuple[str, ...]) -> list[dict[str, object]]:
    """Construit un dictionnaire de caractéristiques pour chaque token."""

    return [token_features(tokens, index) for index in range(len(tokens))]
