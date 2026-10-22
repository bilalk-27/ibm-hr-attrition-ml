
# ============================================================
# 1. IMPORTATION DES LIBRAIRIES
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. CHARGEMENT DES DONNEES
# ============================================================

df = pd.read_csv("kaggle_ibm_dataset_attriction.csv")

print("Dimensions du dataset :", df.shape)


# ============================================================
# 3. ANALYSE EXPLORATOIRE (EDA) - TERMINEE
# ============================================================

# Nous avons déjà étudié :
# - Les valeurs manquantes et les doublons
# - La répartition de la variable Attrition
# - Les distributions de l'âge et du salaire
# - L'attrition selon différentes catégories
# - Les corrélations entre variables

# Cette étape est terminée.
# Nous conservons les analyses précédentes dans Git.


# ============================================================
# 4. SELECTION DES VARIABLES - ACTIF
# ============================================================

variables = [
    # 8 variables initiales
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "OverTime",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobRole",
    "BusinessTravel",

    # 6 nouvelles variables
    "JobLevel",
    "WorkLifeBalance",
    "DistanceFromHome",
    "StockOptionLevel",
    "TotalWorkingYears",
    "YearsSinceLastPromotion"
]

# X contient les variables explicatives
X = df[variables].copy()

# y contient la variable à prédire :
# 1 = départ, 0 = pas de départ
y = df["Attrition"].map({"Yes": 1, "No": 0})

print("Dimensions de X :", X.shape)


# ============================================================
# 5. ENCODAGE - GERE PAR LE PIPELINE
# ============================================================

# Ancienne méthode :
#
# X["OverTime"] = X["OverTime"].map({"Yes": 1, "No": 0})
#
# X = pd.get_dummies(
#     X,
#     columns=["JobRole", "BusinessTravel"],
#     dtype=int
# )
#
# Nous n'utilisons plus cet encodage manuel.
# Le Pipeline s'occupera de transformer les catégories.


# ============================================================
# 6. SEPARATION TRAIN / VALIDATION / TEST
# ============================================================

# Première séparation :
# 80 % pour le développement, 20 % pour le test final

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Deuxième séparation :
# 75 % du développement pour l'entraînement
# 25 % du développement pour la validation

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

print("\n=== REPARTITION DES DONNEES ===")
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

# X_test et y_test ne seront utilisés qu'à la fin du projet.


# ============================================================
# 7. STANDARDISATION - INACTIVE
# ============================================================

# Nous avions utilisé StandardScaler pour la régression
# logistique.
#
# La Random Forest n'a pas besoin de standardisation.
#
# Nous n'utilisons donc pas StandardScaler ici.


# ============================================================
# 8. REGRESSION LOGISTIQUE - INACTIVE
# ============================================================

# Modèle étudié précédemment.
# Nous ne le réentraînons pas pour cet exercice.


# ============================================================
# 9. ANCIENNES RANDOM FOREST - INACTIVES
# ============================================================

# Nous avons déjà testé :
#
# Random Forest 1 :
# - 100 arbres
# - class_weight="balanced"
#
# Random Forest 2 :
# - 100 arbres
# - max_depth=5
# - min_samples_leaf=3
# - class_weight="balanced"
#
# Puis GridSearchCV pour optimiser les hyperparamètres.
#
# Résultats précédents :
#
# 8 variables :
# Meilleur F1-score validation = 0.527
#
# 14 variables :
# Meilleur F1-score validation = 0.512
#
# Nous allons maintenant utiliser un Pipeline.


# ============================================================
# 10. CREATION DU COLUMNTRANSFORMER - ACTIF
# ============================================================

# Colonnes contenant des catégories

colonnes_categorielles = [
    "JobRole",
    "BusinessTravel",
    "OverTime"
]

# OneHotEncoder transforme les catégories en colonnes
# numériques.
#
# handle_unknown="ignore" permet de gérer les catégories
# inconnues lors des futures prédictions.
#
# remainder="passthrough" conserve les colonnes numériques.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "encodage",
            OneHotEncoder(handle_unknown="ignore"),
            colonnes_categorielles
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 11. CREATION DU PIPELINE RANDOM FOREST - ACTIF
# ============================================================

# Le Pipeline regroupe :
#
# Etape 1 : prétraitement des données
# Etape 2 : entraînement du modèle

pipeline_rf = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

from sklearn.model_selection import cross_val_predict

proba_cv = cross_val_predict(
    pipeline_rf,
    X_train,
    y_train,
    cv=cv,
    method='predict_proba',
    n_jobs=-1
)[:, 1]

# print("Nombre de probabilités :", len(proba_cv))
# print("10 premières probabilités :", proba_cv[:10])

seuils = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

for seuil in seuils:
    predictions = (proba_cv >= seuil).astype(int)
    score = f1_score(y_train, predictions)

    print(f"Seuil : {seuil:.0%} | F1-score : {score:.3f}")

# Variables initiales
variables_8 = variables[:8]

# Pipeline avec 8 variables
scores_8 = cross_val_score(
    pipeline_rf,
    X_train[variables_8],
    y_train,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

# Pipeline avec 14 variables
scores_14 = cross_val_score(
    pipeline_rf,
    X_train,
    y_train,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

# print("\n=== COMPARAISON DES MODELES ===")
# print(f"8 variables : {scores_8.mean():.3f}")
# print(f"14 variables : {scores_14.mean():.3f}")

scores_cv = cross_val_score(
    pipeline_rf,
    X_train,
    y_train,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

# print("F1-scores :", scores_cv)
# print("F1-score moyen :", scores_cv.mean())
# print("Écart-type :", scores_cv.std())


# ============================================================
# 12. ENTRAINEMENT DU PIPELINE - ACTIF
# ============================================================

# Le Pipeline apprend les catégories uniquement sur X_train,
# puis entraîne la Random Forest.

pipeline_rf.fit(X_train, y_train)

# print("\nPipeline entraîné avec succès.")


# ============================================================
# 13. PREDICTIONS SUR LA VALIDATION - ACTIF
# ============================================================

# predict_proba renvoie deux probabilités :
# colonne 0 : probabilité de ne pas partir
# colonne 1 : probabilité de partir
#
# [:, 1] récupère la probabilité de départ.

proba_pipeline = pipeline_rf.predict_proba(X_val)[:, 1]

# Seuil de décision de 45 %
#
# Si probabilité >= 0.45 : prédiction = 1 (départ)
# Sinon : prédiction = 0 (pas de départ)

seuil = 0.45

predictions = (proba_pipeline >= seuil).astype(int)


# ============================================================
# 14. EVALUATION DU PIPELINE - ACTIF
# ============================================================

score = f1_score(y_val, predictions)

# print("\n=== RESULTATS DU PIPELINE ===")

# print(f"Seuil utilisé : {seuil:.0%}")
# print(f"F1-score : {score:.3f}")

# print("\nMatrice de confusion :")
# print(confusion_matrix(y_val, predictions))

# print("\nRapport de classification :")
# print(classification_report(y_val, predictions))


# ============================================================
# 15. COMPARAISON AVEC LES RESULTATS PRECEDENTS
# ============================================================

# Ces valeurs ont été obtenues lors des exercices précédents.
# Elles servent de points de comparaison.

f1_initial = 0.527
f1_enrichi = 0.512

# print("\n=== COMPARAISON ===")

# print(f"Ancien modèle (8 variables) : {f1_initial:.3f}")
# print(f"Ancien modèle (14 variables) : {f1_enrichi:.3f}")
# print(f"Nouveau Pipeline (14 variables) : {score:.3f}")

# ============================================================
# 17. VERIFICATION DU SEUIL SUR X_VAL
# ============================================================

# Seuil sélectionné grâce à la validation croisée
seuil_final = 0.50

# Probabilités calculées par notre Pipeline
proba_val = pipeline_rf.predict_proba(X_val)[:, 1]

# Transformation des probabilités en prédictions
predictions_val = (proba_val >= seuil_final).astype(int)

# Évaluation
score_val = f1_score(y_val, predictions_val)

# print("\n=== VALIDATION DU SEUIL 50 % ===")
# print(f"F1-score : {score_val:.3f}")

# print("\nMatrice de confusion :")
# print(confusion_matrix(y_val, predictions_val))

# print("\nRapport de classification :")
# print(classification_report(y_val, predictions_val))


# ============================================================
# 18. ANALYSE DES ERREURS
# ============================================================

# Copier les données de validation
analyse_erreurs = X_val.copy()

# Ajouter les valeurs réelles et prédites
analyse_erreurs["Attrition_reelle"] = y_val
analyse_erreurs["Attrition_predite"] = predictions_val

# Ajouter la probabilité estimée de départ
analyse_erreurs["Probabilite_depart"] = proba_val

# Faux négatifs :
# collaborateurs partis mais prédits comme restant
faux_negatifs = analyse_erreurs[
    (analyse_erreurs["Attrition_reelle"] == 1) &
    (analyse_erreurs["Attrition_predite"] == 0)
]

# Faux positifs :
# collaborateurs restés mais prédits comme partant
faux_positifs = analyse_erreurs[
    (analyse_erreurs["Attrition_reelle"] == 0) &
    (analyse_erreurs["Attrition_predite"] == 1)
]

# print("\n=== ANALYSE DES ERREURS ===")

# print("Nombre de faux négatifs :", len(faux_negatifs))
# print("Nombre de faux positifs :", len(faux_positifs))

# print("\n=== FAUX NEGATIFS ===")
# print(
#     faux_negatifs[
#         ["Age", "MonthlyIncome", "OverTime", "Probabilite_depart"]
#     ].head(10)
# )


# ============================================================
# 16. PROCHAINES ETAPES - A VENIR
# ============================================================

# 1. Comprendre et vérifier les résultats du Pipeline
# 2. Utiliser la validation croisée avec le Pipeline
# 3. Comparer les modèles plus rigoureusement
# 4. Choisir un seuil de décision
# 5. Réentraîner le modèle final sur X_dev et y_dev
# 6. Evaluer une seule fois sur X_test et y_test
# 7. Sauvegarder le modèle avec joblib
# 8. Préparer le README GitHub

# ============================================================
# 19. ENTRAINEMENT FINAL
# ============================================================

from sklearn.base import clone

# Créer une nouvelle copie du Pipeline
pipeline_final = clone(pipeline_rf)

# Entraîner sur toutes les données de développement
pipeline_final.fit(X_dev, y_dev)

# print("Modèle final entraîné avec succès.")
# print("Nombre de collaborateurs :", len(X_dev))

# ============================================================
# 20. EVALUATION FINALE SUR X_TEST
# ============================================================

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Calculer les probabilités de départ
proba_test = pipeline_final.predict_proba(X_test)[:, 1]

# 2. Appliquer notre seuil fixé à 50 %
predictions_test = (proba_test >= 0.50).astype(int)

# 3. Calculer les métriques
f1 = f1_score(y_test, predictions_test)
precision = precision_score(y_test, predictions_test)
recall = recall_score(y_test, predictions_test)
accuracy = accuracy_score(y_test, predictions_test)

# 4. Afficher les résultats
print("\n=== EVALUATION FINALE ===")

print(f"F1-score : {f1:.3f}")
print(f"Précision : {precision:.3f}")
print(f"Rappel : {recall:.3f}")
print(f"Accuracy : {accuracy:.3f}")

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, predictions_test))

print("\nRapport de classification :")
print(classification_report(y_test, predictions_test))

# ============================================================
# 21. SAUVEGARDE DU MODELE FINAL
# ============================================================

import joblib
from pathlib import Path

# Créer un dossier pour stocker le modèle
dossier_modeles = Path("models")
dossier_modeles.mkdir(exist_ok=True)

# Définir le chemin de sauvegarde
chemin_modele = dossier_modeles / "random_forest_attrition.joblib"

# Sauvegarder le Pipeline entraîné
joblib.dump(pipeline_final, chemin_modele)

print(f"\nModèle sauvegardé : {chemin_modele}")

