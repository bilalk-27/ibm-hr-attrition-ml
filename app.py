
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="📊",
    layout="centered"
)

# ============================================================
# 2. CHARGEMENT DU MODELE
# ============================================================

@st.cache_resource
def charger_modele():
    chemin = Path(__file__).resolve().parent / "models" / "random_forest_attrition.joblib"
    return joblib.load(chemin)

modele = charger_modele()

# ============================================================
# 3. TITRE
# ============================================================

st.title("HR Attrition Predictor")

st.write(
    "Application pédagogique de Machine Learning "
    "pour estimer le départ d'un collaborateur fictif."
)

st.info(
    "Ce simulateur utilise un jeu de données synthétique. "
    "Il ne doit pas servir à prendre des décisions RH réelles."
)

st.divider()

# ============================================================
# 4. INFORMATIONS DU COLLABORATEUR
# ============================================================

st.subheader("Informations personnelles et professionnelles")

age = st.slider("Âge", 18, 65, 28)

salaire = st.number_input(
    "Salaire mensuel",
    min_value=1000,
    max_value=20000,
    value=3500,
    step=100
)

anciennete = st.slider(
    "Ancienneté dans l'entreprise",
    0, 40, 2
)

heures_sup = st.selectbox(
    "Heures supplémentaires",
    ["Yes", "No"]
)

poste = st.selectbox(
    "Poste occupé",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ]
)

deplacements = st.selectbox(
    "Déplacements professionnels",
    [
        "Non-Travel",
        "Travel_Rarely",
        "Travel_Frequently"
    ]
)

niveau_poste = st.slider(
    "Niveau du poste",
    1, 5, 2
)

distance = st.slider(
    "Distance domicile-travail",
    1, 29, 15
)

# ============================================================
# 5. SATISFACTION ET CONDITIONS DE TRAVAIL
# ============================================================

st.divider()

st.subheader("Satisfaction et conditions de travail")

satisfaction = st.slider(
    "Satisfaction professionnelle",
    1, 4, 2
)

environnement = st.slider(
    "Satisfaction de l'environnement de travail",
    1, 4, 2
)

equilibre = st.slider(
    "Équilibre vie professionnelle / personnelle",
    1, 4, 2
)

stock_options = st.slider(
    "Niveau de stock-options",
    0, 3, 0
)

experience_totale = st.slider(
    "Années d'expérience professionnelle",
    0, 40, 5
)

derniere_promotion = st.slider(
    "Années depuis la dernière promotion",
    0, 15, 1
)

# ============================================================
# 6. CONSTRUCTION DU DATAFRAME
# ============================================================

collaborateur = pd.DataFrame([{
    "Age": age,
    "MonthlyIncome": salaire,
    "YearsAtCompany": anciennete,
    "OverTime": heures_sup,
    "JobSatisfaction": satisfaction,
    "EnvironmentSatisfaction": environnement,
    "JobRole": poste,
    "BusinessTravel": deplacements,
    "JobLevel": niveau_poste,
    "WorkLifeBalance": equilibre,
    "DistanceFromHome": distance,
    "StockOptionLevel": stock_options,
    "TotalWorkingYears": experience_totale,
    "YearsSinceLastPromotion": derniere_promotion
}])

# ============================================================
# 7. PREDICTION
# ============================================================

st.divider()

if st.button("Prédire le départ", type="primary"):

    probabilite = modele.predict_proba(collaborateur)[0, 1]

    prediction = int(probabilite >= 0.50)

    st.subheader("Résultat de la prédiction")

    st.metric(
        "Score estimé de départ",
        f"{probabilite:.2%}"
    )

    if prediction == 1:
        st.warning("Classe prédite : Départ")
    else:
        st.success("Classe prédite : Pas de départ")

    st.caption(
        "Seuil de classification : 50 %. "
        "Le score n'est pas une probabilité calibrée "
        "ni une certitude concernant une personne."
    )

# ============================================================
# 8. AFFICHAGE DES DONNEES
# ============================================================

with st.expander("Voir les données envoyées au modèle"):
    st.dataframe(collaborateur)
