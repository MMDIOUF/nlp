"""Application Streamlit autonome pour le modèle NER wolof."""

from html import escape
from pathlib import Path
import re

import joblib
import streamlit as st


MODEL_PATH = Path("crf_wolof.joblib")
COLORS = {
    "PER": "#dbeafe",
    "LOC": "#dcfce7",
    "ORG": "#fef3c7",
    "DATE": "#fce7f3",
}
SENEGAL_LOCATIONS = {
    "dakar", "thiès", "thies", "saint-louis", "kaolack", "ziguinchor",
    "touba", "rufisque", "diourbel", "louga", "fatick", "kolda",
    "matam", "kédougou", "kedougou", "sédhiou", "sedhiou", "tambacounda",
}
MONTHS = {
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "août", "aout", "septembre", "octobre", "novembre",
    "décembre", "decembre",
}


def word_shape(word):
    return "".join(
        "X" if char.isupper() else
        "x" if char.islower() else
        "d" if char.isdigit() else char
        for char in word
    )


def token_features(tokens, index):
    word = tokens[index]
    lower = word.lower()
    features = {
        "bias": 1.0,
        "word.lower": lower,
        "word.prefix1": lower[:1],
        "word.prefix2": lower[:2],
        "word.suffix2": lower[-2:],
        "word.suffix3": lower[-3:],
        "word.shape": word_shape(word),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
    }
    if index == 0:
        features["BOS"] = True
    else:
        previous = tokens[index - 1]
        features.update({
            "-1:word.lower": previous.lower(),
            "-1:word.istitle": previous.istitle(),
            "-1:word.isupper": previous.isupper(),
        })
    if index == len(tokens) - 1:
        features["EOS"] = True
    else:
        following = tokens[index + 1]
        features.update({
            "+1:word.lower": following.lower(),
            "+1:word.istitle": following.istitle(),
            "+1:word.isupper": following.isupper(),
        })
    return features


def sentence_features(tokens):
    return [token_features(tokens, index) for index in range(len(tokens))]


def tokenize_text(text):
    return re.findall(r"[^\W_]+(?:[-'][^\W_]+)*|[^\w\s]", text, flags=re.UNICODE)


def assisted_labels(tokens, crf_labels):
    labels = list(crf_labels)
    for index, token in enumerate(tokens):
        if labels[index] == "O" and token.lower() in SENEGAL_LOCATIONS:
            labels[index] = "B-LOC"
        if labels[index] == "O" and token.isupper() and token.isalpha() and len(token) >= 2:
            labels[index] = "B-ORG"

    previous = -2
    for index, token in enumerate(tokens):
        if token.isdigit() or token.lower() in MONTHS:
            if labels[index] == "O" or labels[index].endswith("-PER"):
                labels[index] = "I-DATE" if index == previous + 1 else "B-DATE"
            previous = index

    index = 0
    while index + 1 < len(tokens):
        if (
            tokens[index].istitle()
            and tokens[index + 1].istitle()
            and labels[index] == "O"
            and labels[index + 1] == "O"
        ):
            labels[index:index + 2] = ["B-PER", "I-PER"]
            index += 2
        else:
            index += 1
    return labels


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(text, assisted):
    tokens = tokenize_text(text)
    labels = list(load_model().predict_single(sentence_features(tokens)))
    return tokens, assisted_labels(tokens, labels) if assisted else labels


def highlighted_text(tokens, labels):
    parts = []
    for token, label in zip(tokens, labels):
        entity_type = label.split("-", 1)[1] if "-" in label else None
        if entity_type:
            parts.append(
                f'<span style="background:{COLORS[entity_type]};padding:5px 8px;'
                f'border-radius:7px;margin:3px;display:inline-block">'
                f'{escape(token)} <small><b>{entity_type}</b></small></span>'
            )
        else:
            parts.append(f'<span style="padding:5px 3px">{escape(token)}</span>')
    return " ".join(parts)


st.set_page_config(page_title="NER Wolof - CRF", page_icon="🇸🇳", layout="centered")
st.title("Reconnaissance d'entités nommées en wolof")
st.caption("Modèle CRF entraîné sur le corpus Wolof de MasakhaNER")

with st.expander("Comment lire le résultat ?"):
    st.write("**PER** = personne, **LOC** = lieu, **ORG** = organisation, **DATE** = date.")

text = st.text_area(
    "Écrivez une phrase en wolof",
    value="Maki Sàll dem na Dakar.",
    height=120,
)
assisted = st.toggle(
    "Mode assisté : CRF + règles pour les entités nouvelles",
    value=True,
)

if st.button("Détecter les entités", type="primary", width="stretch"):
    if not text.strip():
        st.warning("Veuillez écrire une phrase.")
    else:
        tokens, labels = predict(text.strip(), assisted)
        st.caption(
            "CRF complété par des règles explicables."
            if assisted else "Résultat brut du modèle CRF."
        )
        st.subheader("Texte annoté")
        st.markdown(highlighted_text(tokens, labels), unsafe_allow_html=True)
        st.subheader("Labels BIO")
        st.dataframe(
            [{"Token": token, "Label prédit": label} for token, label in zip(tokens, labels)],
            width="stretch",
            hide_index=True,
        )

st.info(
    "Limite : une entité rare ou inconnue peut être manquée par le CRF brut."
)
