"""Mini-application Streamlit pour tester le modèle NER wolof."""

from html import escape
from pathlib import Path

import joblib
import streamlit as st

from src.features import sentence_features
from src.postprocess import assisted_labels
from src.predict import tokenize_text


MODEL_PATH = Path("models/crf_wolof.joblib")
COLORS = {
    "PER": "#dbeafe",
    "LOC": "#dcfce7",
    "ORG": "#fef3c7",
    "DATE": "#fce7f3",
}


@st.cache_resource
def load_model():
    """Charger le modèle une seule fois pour accélérer les prédictions."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modèle absent. Exécutez d'abord : python -m src.finalize_crf"
        )
    return joblib.load(MODEL_PATH)


def predict(text: str, assisted: bool) -> tuple[list[str], list[str]]:
    """Transformer le texte en tokens puis prédire un label BIO par token."""
    tokens = tokenize_text(text)
    model = load_model()
    labels = list(model.predict_single(sentence_features(tokens)))
    if assisted:
        labels = assisted_labels(tokens, labels)
    return tokens, labels


def highlighted_text(tokens: list[str], labels: list[str]) -> str:
    """Colorer les tokens reconnus et afficher leur type d'entité."""
    parts = []
    for token, label in zip(tokens, labels):
        entity_type = label.split("-", 1)[1] if "-" in label else None
        if entity_type:
            color = COLORS[entity_type]
            parts.append(
                f'<span style="background:{color};padding:5px 8px;'
                f'border-radius:7px;margin:3px;display:inline-block">'
                f'{escape(token)} <small><b>{entity_type}</b></small></span>'
            )
        else:
            parts.append(
                f'<span style="padding:5px 3px;display:inline-block">'
                f'{escape(token)}</span>'
            )
    return " ".join(parts)


st.set_page_config(page_title="NER Wolof - CRF", page_icon="🇸🇳", layout="centered")

st.title("Reconnaissance d'entités nommées en wolof")
st.caption("Modèle CRF entraîné sur les données Wolof de MasakhaNER")

with st.expander("Comment lire le résultat ?"):
    st.write(
        "**PER** = personne, **LOC** = lieu, **ORG** = organisation, "
        "**DATE** = date. Le modèle attribue d'abord un label BIO à chaque token."
    )

text = st.text_area(
    "Écrivez une phrase en wolof",
    value="Maki Sàll dem na Dakar.",
    height=120,
)

assisted = st.toggle(
    "Mode assisté : CRF + règles pour les entités nouvelles",
    value=True,
    help="Ajoute des règles simples pour les noms complets, sigles, villes et dates.",
)

if st.button("Détecter les entités", type="primary", width="stretch"):
    if not text.strip():
        st.warning("Veuillez écrire une phrase avant de lancer la détection.")
    else:
        try:
            tokens, labels = predict(text.strip(), assisted=assisted)
            st.caption(
                "Résultat du CRF complété par des règles explicables."
                if assisted
                else "Résultat brut du modèle CRF évalué dans le notebook."
            )
            st.subheader("Texte annoté")
            st.markdown(highlighted_text(tokens, labels), unsafe_allow_html=True)

            st.subheader("Labels BIO")
            st.dataframe(
                [{"Token": token, "Label prédit": label} for token, label in zip(tokens, labels)],
                width="stretch",
                hide_index=True,
            )
        except Exception as error:
            st.error(f"Impossible de lancer la prédiction : {error}")

st.info(
    "Limite : le CRF utilise surtout la forme des mots et leur voisinage local. "
    "Une entité rare ou inconnue peut donc être manquée."
)
