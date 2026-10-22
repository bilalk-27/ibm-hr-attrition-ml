import joblib

modele = joblib.load("models/random_forest_attrition.joblib")

print("Modèle chargé avec succès !")

import pandas as pd

nouveau_collaborateur = pd.DataFrame([{
    "Age": 28,
    "MonthlyIncome": 3500,
    "YearsAtCompany": 2,
    "OverTime": "Yes",
    "JobSatisfaction": 2,
    "EnvironmentSatisfaction": 2,
    "JobRole": "Sales Executive",
    "BusinessTravel": "Travel_Frequently",
    "JobLevel": 2,
    "WorkLifeBalance": 2,
    "DistanceFromHome": 15,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 5,
    "YearsSinceLastPromotion": 1
}])

# Calculer la probabilité estimée de départ
probabilite = modele.predict_proba(nouveau_collaborateur)[0, 1]

# Appliquer notre seuil de décision
prediction = int(probabilite >= 0.50)

# Copier notre collaborateur initial
collaborateur_b = nouveau_collaborateur.copy()

# Modifier uniquement les heures supplémentaires
collaborateur_b["OverTime"] = "No"

# Calculer la nouvelle probabilité
proba_b = modele.predict_proba(collaborateur_b)[0, 1]

print("\n=== COMPARAISON DES PROFILS ===")
print(f"Profil A (OverTime = Yes) : {probabilite:.2%}")
print(f"Profil B (OverTime = No)  : {proba_b:.2%}")