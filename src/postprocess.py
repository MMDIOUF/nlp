"""Règles prudentes pour compléter le CRF dans l'application de démonstration."""

SENEGAL_LOCATIONS = {
    "dakar", "thiès", "thies", "saint-louis", "kaolack", "ziguinchor",
    "touba", "rufisque", "diourbel", "louga", "fatick", "kolda",
    "matam", "kédougou", "kedougou", "sédhiou", "sedhiou", "tambacounda",
}

MONTHS = {
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "août", "aout", "septembre", "octobre", "novembre", "décembre",
    "decembre",
}


def assisted_labels(tokens: list[str], crf_labels: list[str]) -> list[str]:
    """Complète seulement les prédictions O avec quelques règles explicables."""

    labels = list(crf_labels)

    # Villes sénégalaises connues.
    for index, token in enumerate(tokens):
        if labels[index] == "O" and token.lower() in SENEGAL_LOCATIONS:
            labels[index] = "B-LOC"

    # Sigles comme ONU, RTS ou UCAD.
    for index, token in enumerate(tokens):
        if labels[index] == "O" and token.isupper() and token.isalpha() and len(token) >= 2:
            labels[index] = "B-ORG"

    # Années, nombres et noms de mois.
    date_indexes = [
        index for index, token in enumerate(tokens)
        if token.isdigit() or token.lower() in MONTHS
    ]
    previous = -2
    for index in date_indexes:
        if labels[index] == "O" or labels[index].endswith("-PER"):
            labels[index] = "I-DATE" if index == previous + 1 else "B-DATE"
        previous = index

    # Deux mots title-case consécutifs constituent souvent un nom complet.
    index = 0
    while index + 1 < len(tokens):
        if (
            tokens[index].istitle()
            and tokens[index + 1].istitle()
            and labels[index] == "O"
            and labels[index + 1] == "O"
        ):
            labels[index] = "B-PER"
            labels[index + 1] = "I-PER"
            index += 2
        else:
            index += 1

    return labels
