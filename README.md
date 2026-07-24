# NER wolof avec un CRF

Ce projet reconnaît quatre types d'entités dans une phrase wolof :

- `PER` : personne ;
- `LOC` : lieu ;
- `ORG` : organisation ;
- `DATE` : date.

Le modèle principal est un **CRF linéaire**. Il est simple, rapide et interprétable : il combine les indices observés sur chaque mot avec les transitions possibles entre les labels BIO.

Application : https://5aaxfh5jf9kkjgjna7fpso.streamlit.app/

Guide visuel de soutenance : [ouvrir le README en PDF](output/pdf/README.pdf)

## Résultat essentiel

| Étape | Résultat |
|---|---:|
| Baseline de mémorisation, F1 dev | 0,6811 |
| CRF sélectionné, F1 dev | 0,7583 |
| CRF final, F1 test strict | 0,6622 |

Le résultat test est calculé une seule fois après le choix des hyperparamètres sur `dev`. La baisse entre dev et test montre honnêtement la difficulté de généralisation, surtout pour `DATE` et `ORG`.

## Pipeline

```text
Corpus CoNLL
   ↓
Tokens + labels BIO
   ↓
Caractéristiques de chaque token
   ↓
Entraînement du CRF sur train
   ↓
Choix de c1/c2 sur dev
   ↓
Évaluation finale unique sur test
   ↓
Modèle sauvegardé + application Streamlit
```

## Organisation du projet

| Élément | Rôle |
|---|---|
| `projet_nlp.ipynb` | Démonstration progressive et preuves du raisonnement |
| `data/raw/` | Corpus `train`, `dev` et `test` au format CoNLL |
| `src/analyze_data.py` | Chargement, validation BIO et statistiques |
| `src/features.py` | Caractéristiques observées par le CRF |
| `src/dataset.py` | Construction de `X` et `y` |
| `src/train_crf.py` | Configuration et entraînement |
| `src/evaluate.py` | Précision, rappel et F1 strict IOB2 |
| `src/error_analysis.py` | Classification des vraies erreurs |
| `src/select_crf.py` | Comparaison limitée des hyperparamètres sur dev |
| `src/finalize_crf.py` | Entraînement final, test unique et sauvegarde |
| `src/baseline.py` | Point de comparaison par mémorisation |
| `src/predict.py` | Tokenisation et prédiction sur une nouvelle phrase |
| `src/postprocess.py` | Règles explicables du mode assisté |
| `app.py` | Interface Streamlit |
| `models/` | Modèle entraîné et métadonnées finales |

## Repères dans le notebook

| Cellules | Ce que je démontre | Fichiers associés |
|---|---|---|
| 3-6 | Imports, chargement de train et dev | `analyze_data.py` |
| 7-11 | Labels BIO et déséquilibre | `analyze_data.py` |
| 12-13 | Construction alignée de X et y | `dataset.py`, `features.py` |
| 14-17 | Entraînement et transitions apprises | `train_crf.py` |
| 18-22 | Prédictions et F1 strict sur dev | `evaluate.py` |
| 23-25 | Analyse des erreurs réelles | `error_analysis.py` |
| 26-28 | Sélection de c1 et c2 sur dev | `select_crf.py` |
| 29-31 | Résultat final unique sur test | `finalize_crf.py` |
| 32-33 | Prédiction sur un nouveau texte | `predict.py` |
| 34-36 | Comparaison avec la baseline | `baseline.py` |
| 37-39 | Architecture, limites et conclusion | ensemble du projet |

## Les idées à savoir défendre

### Pourquoi BIO ?

`B-TYPE` commence une entité, `I-TYPE` la continue et `O` signifie « hors entité ». Par exemple :

```text
Maki  B-PER
Sàll  I-PER
dem   O
Dakar B-LOC
```

L'alignement est obligatoire : chaque token possède exactement un label.

### Que contient X ? Que contient y ?

- `X` contient une séquence de dictionnaires de caractéristiques ;
- `y` contient la séquence des labels BIO attendus ;
- `len(X[i]) == len(y[i])` pour chaque phrase.

Les caractéristiques incluent le mot en minuscules, ses préfixes, ses suffixes, sa forme, la casse, les chiffres et le voisinage immédiat.

### Que fait le CRF ?

Le CRF choisit la meilleure séquence complète de labels. Il combine :

1. les **émissions**, c'est-à-dire la compatibilité entre les caractéristiques d'un token et un label ;
2. les **transitions**, c'est-à-dire la compatibilité entre deux labels voisins ;
3. la **régularisation**, qui limite le surapprentissage.

Configuration finale :

```python
algorithm="lbfgs"
c1=0.05
c2=0.20
max_iterations=100
all_possible_transitions=True
```

### Pourquoi séparer train, dev et test ?

- `train` apprend les poids du CRF ;
- `dev` compare les configurations ;
- `test` mesure une seule fois la généralisation après le choix final.

Utiliser test pour choisir le modèle rendrait le résultat final optimiste et non fiable.

### Pourquoi le F1 strict ?

Une entité est correcte seulement si son **type** et ses **limites** sont exacts. `seqeval` est donc utilisé avec `mode="strict"` et `scheme=IOB2`.

- précision : parmi les entités prédites, combien sont correctes ?
- rappel : parmi les entités attendues, combien sont retrouvées ?
- F1 : équilibre entre précision et rappel.

### Pourquoi le mode assisté ?

Le résultat scientifique du notebook est le **CRF brut**. Dans l'application, le mode assisté peut compléter les labels `O` avec des règles simples pour certaines villes, sigles, dates et noms composés. Il sert à la démonstration, mais ne remplace pas l'évaluation du CRF.

## Exécution

```powershell
python -m pip install -r requirements.txt
python -m src.analyze_data
python -m src.train_crf
python -m src.predict --text "Maki Sàll dem Dakar"
streamlit run app.py
```

Le notebook peut être lancé avec Jupyter :

```powershell
jupyter notebook projet_nlp.ipynb
```

## Limites et amélioration possible

Le CRF utilise surtout la forme des mots et leur contexte local. Il peut manquer une entité rare ou inconnue. Une suite logique serait de comparer ce modèle interprétable à un transformeur multilingue ou africain préentraîné, tout en gardant exactement le même protocole train/dev/test et la même évaluation stricte.
