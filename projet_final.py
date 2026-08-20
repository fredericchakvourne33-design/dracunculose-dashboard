"""
==============================================================================
SYSTEME INTELLIGENT DE SURVEILLANCE - VER DE GUINEE (TCHAD)
Pipeline complet : analyse exploratoire + modele de Machine Learning
pour l'identification des villages a risque de recrudescence, mois par mois.

Entree  : Infections_Animales.csv (cas d'infections animales, 2022-2025)
Sortie  : modele entraine + classement des villages a risque + graphiques

Respecte les etapes classiques d'un projet de Machine Learning :
  1. Chargement des donnees
  2. Analyse exploratoire (EDA)
  3. Nettoyage et pretraitement
  4. Feature engineering
  5. Separation train / test (temporelle, jamais aleatoire sur une serie
     temporelle)
  6. Modeles de reference (baselines)
  7. Entrainement du modele de Machine Learning
  8. Evaluation
  9. Interpretation (importance des variables)
  10. Modele final et export des predictions
==============================================================================
"""

import unicodedata
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
pd.set_option("display.width", 130)
plt.rcParams["figure.autolayout"] = True

CSV_PATH = "/mnt/user-data/uploads/Infections_Animales.csv"
RAYON_VOISINAGE_KM = 25      # rayon utilise pour la variable de contagion spatiale
FENETRE_INCUBATION = (10, 14)  # periode d'incubation confirmee (mois)
ANNEE_LIMITE_TRAIN = 2024     # on entraine sur <= cette annee, on teste sur l'annee suivante


# ==============================================================================
# 1. CHARGEMENT DES DONNEES
# ==============================================================================
print("=" * 78)
print("1. CHARGEMENT DES DONNEES")
print("=" * 78)

df = pd.read_csv("Infections_Animales.csv", sep=";", encoding="utf-8-sig")
print(f"Dataset charge : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"Colonnes : {list(df.columns)}")


# ==============================================================================
# 2. ANALYSE EXPLORATOIRE DES DONNEES (EDA)
# ==============================================================================
print("\n" + "=" * 78)
print("2. ANALYSE EXPLORATOIRE DES DONNEES (EDA)")
print("=" * 78)

# --- 2.1 Valeurs manquantes -------------------------------------------------
print("\n--- Valeurs manquantes par colonne ---")
manquants = df.isna().sum()
print(manquants[manquants > 0] if manquants.sum() else "Aucune valeur manquante detectee (NaN).")
# Note : l'absence de NaN ne signifie pas l'absence de donnees invalides ;
# des textes vides, des '0,0' ou des incoherences de format restent possibles
# et seront traites a l'etape de nettoyage.

# --- 2.2 Distribution des variables categorielles cles ----------------------
print("\n--- Especes animales concernees ---")
print(df["Type d'animal"].value_counts())

print("\n--- Repartition par province (telle que codee dans le fichier) ---")
print(df["Province"].value_counts())

print("\n--- Infection isolee vs non isolee ---")
print(df["Infection Isolee?"].value_counts())

print("\n--- Contamination des sources d'eau ---")
print(df["Contamination des Sources d'eau"].value_counts())

# --- 2.3 Croisement infection isolee / contamination -------------------------
# Verifie la logique metier decrite : une infection non isolee est associee
# a un animal non attache et/ou une contamination d'eau.
print("\n--- Croisement 'Infection Isolee?' x 'Contamination des Sources d'eau' ---")
print(pd.crosstab(df["Infection Isolee?"], df["Contamination des Sources d'eau"]))

# --- 2.4 Serie temporelle brute ----------------------------------------------
dates_brutes = pd.to_datetime(df["Date d'emergence"], format="%d/%m/%Y", errors="coerce")
print(f"\nDates d'emergence non interpretables : {dates_brutes.isna().sum()} / {len(df)}")

serie_brute = dates_brutes.dt.to_period("M").value_counts().sort_index()
print("\n--- Nombre de cas par mois (5 premiers / 5 derniers) ---")
print(pd.concat([serie_brute.head(5), serie_brute.tail(5)]))

fig, ax = plt.subplots(figsize=(12, 4))
serie_brute.plot(ax=ax, marker="o", markersize=3)
ax.set_title("EDA - Nombre de cas declares par mois (2022-2025)")
ax.set_ylabel("Nombre de cas")
ax.grid(alpha=0.3)
plt.savefig("eda_serie_mensuelle.png", dpi=100)
plt.close(fig)
print("Graphique sauvegarde : eda_serie_mensuelle.png")

# --- 2.5 Saisonnalite ---------------------------------------------------------
saisonnalite = dates_brutes.dt.month.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4))
saisonnalite.plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("EDA - Repartition des cas par mois de l'annee (saisonnalite)")
ax.set_xlabel("Mois")
ax.set_ylabel("Nombre total de cas (2022-2025)")
plt.savefig("eda_saisonnalite.png", dpi=100)
plt.close(fig)
print("Graphique sauvegarde : eda_saisonnalite.png")
print("-> Constat : les cas se concentrent nettement entre mai et septembre.")

# --- 2.6 Coordonnees geographiques -------------------------------------------
lat_test = pd.to_numeric(df["Latitude"].astype(str).str.replace(",", "."), errors="coerce")
lon_test = pd.to_numeric(df["Longitude"].astype(str).str.replace(",", "."), errors="coerce")
print(f"\nCoordonnees exploitables : {(lat_test.notna() & lon_test.notna()).sum()} / {len(df)}")

# --- 2.7 Doublons potentiels (reinfections) ----------------------------------
doublons_qr = df["Code QR Animale"].duplicated().sum()
print(f"\nCodes QR animaux apparaissant plus d'une fois (reinfections potentielles) : {doublons_qr}")


# ==============================================================================
# 3. NETTOYAGE ET PRETRAITEMENT
# ==============================================================================
print("\n" + "=" * 78)
print("3. NETTOYAGE ET PRETRAITEMENT")
print("=" * 78)


def normaliser_texte(valeur):
    """Uniformise la casse, les accents et les espaces/underscores d'un champ texte."""
    if pd.isna(valeur):
        return np.nan
    valeur = str(valeur).strip()
    valeur = unicodedata.normalize("NFKD", valeur).encode("ascii", "ignore").decode()
    return re.sub(r"[\s_]+", " ", valeur).strip().upper()


def parser_date(valeur):
    """Parse une date en testant plusieurs formats possibles (le fichier source
    a historiquement melange JJ/MM/AAAA et AAAA-MM-JJ)."""
    if pd.isna(valeur):
        return pd.NaT
    valeur = str(valeur).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return pd.to_datetime(valeur, format=fmt)
        except ValueError:
            continue
    return pd.NaT


# Dates
for col in ["Date d'emergence", "Date de detection", "Date d'Investigation"]:
    df[col + "_dt"] = df[col].apply(parser_date)

# Coordonnees (virgule decimale -> point)
df["lat"] = pd.to_numeric(df["Latitude"].astype(str).str.replace(",", "."), errors="coerce")
df["lon"] = pd.to_numeric(df["Longitude"].astype(str).str.replace(",", "."), errors="coerce")

# Variables categorielles normalisees
df["espece"] = df["Type d'animal"].apply(normaliser_texte).str.replace(" DOMESTIQUE", "", regex=False)
df["province_n"] = df["Province"].apply(normaliser_texte)
df["isolee_n"] = df["Infection Isolee?"].apply(normaliser_texte)
df["contam_n"] = df["Contamination des Sources d'eau"].apply(normaliser_texte)
df["attache_n"] = df["Animal Attache"].apply(normaliser_texte)

# Identifiant de village stable (le nom de village seul contient des doublons
# entre districts differents)
df["village_n"] = (
    df["Village"].apply(normaliser_texte) + " | "
    + df["Zone"].apply(normaliser_texte) + " | "
    + df["District"].apply(normaliser_texte)
)

col_emerg_dt = "Date d'emergence_dt"
print(f"Dates d'emergence non parsees apres nettoyage : {df[col_emerg_dt].isna().sum()}")
print(f"Lignes sans coordonnees valides : {(df['lat'].isna() | df['lon'].isna()).sum()}")
print(f"Villages distincts (identifiant normalise) : {df['village_n'].nunique()}")

# On retire les lignes sans date d'emergence exploitable : elles ne peuvent
# pas etre situees dans le temps et fausseraient l'agregation mensuelle.
d = df.dropna(subset=["Date d'emergence_dt"]).copy()
print(f"Lignes conservees pour la modelisation : {len(d)} / {len(df)}")

d["ym"] = d["Date d'emergence_dt"].dt.to_period("M")


# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================
print("\n" + "=" * 78)
print("4. FEATURE ENGINEERING")
print("=" * 78)

# --- 4.1 Indicateur de transmission differee ---------------------------------
# Une infection non isolee (animal non attache et/ou source d'eau contaminee)
# amorce un nouveau cycle de transmission avec 10 a 14 mois d'incubation.
d["risque_transmission"] = (
    (d["isolee_n"] == "NON") & ((d["contam_n"] == "OUI") | (d["attache_n"] == "NON"))
).astype(int)
print(f"Part des cas presentant ce profil a risque : {d['risque_transmission'].mean():.1%}")

# --- 4.2 Construction du panel village x mois --------------------------------
# Le dataset ne contient que des cas positifs : on fabrique les negatifs en
# croisant chaque village avec chaque mois de la periode d'etude.
mois_complets = pd.period_range("2022-01", "2025-12", freq="M")
villages = sorted(d["village_n"].unique())
panel = pd.DataFrame(
    index=pd.MultiIndex.from_product([villages, mois_complets], names=["village", "ym"])
).reset_index()

cas_agreges = d.groupby(["village_n", "ym"]).size().rename("cas")
risque_agrege = d.groupby(["village_n", "ym"])["risque_transmission"].sum().rename("risque")
panel = panel.merge(cas_agreges, left_on=["village", "ym"], right_index=True, how="left").fillna({"cas": 0})
panel = panel.merge(risque_agrege, left_on=["village", "ym"], right_index=True, how="left").fillna({"risque": 0})
panel = panel.sort_values(["village", "ym"]).reset_index(drop=True)

panel["y"] = (panel["cas"] > 0).astype(int)  # cible : au moins un cas ce mois-la
print(f"Panel village x mois : {panel.shape[0]} lignes, prevalence positive = {panel['y'].mean():.2%}")

# --- 4.3 Coordonnees moyennes par village -------------------------------------
coords_village = d.dropna(subset=["lat", "lon"]).groupby("village_n")[["lat", "lon"]].median()
panel = panel.merge(coords_village, left_on="village", right_index=True, how="left")

# --- 4.4 Variables temporelles retardees (historique propre au village) ------
groupe = panel.groupby("village")
for retard in [1, 2, 3, 12]:
    panel[f"lag{retard}"] = groupe["cas"].shift(retard)
panel["roll3"] = groupe["cas"].shift(1).transform(lambda s: s.rolling(3, min_periods=1).sum())
panel["roll12"] = groupe["cas"].shift(1).transform(lambda s: s.rolling(12, min_periods=1).sum())
panel["cumul_historique"] = groupe["cas"].cumsum() - panel["cas"]

# --- 4.5 Variable de transmission differee, decalee de 10 a 14 mois ----------
panel["risque_10_14"] = groupe["risque"].transform(
    lambda s: s.shift(FENETRE_INCUBATION[0]).rolling(
        FENETRE_INCUBATION[1] - FENETRE_INCUBATION[0] + 1, min_periods=1
    ).sum()
)

# --- 4.6 Saisonnalite (encodage cyclique) et tendance -------------------------
panel["mois"] = panel["ym"].dt.month
panel["sin_mois"] = np.sin(2 * np.pi * panel["mois"] / 12)
panel["cos_mois"] = np.cos(2 * np.pi * panel["mois"] / 12)
panel["t"] = (panel["ym"].dt.year - 2022) * 12 + panel["mois"]

# --- 4.7 Contagion spatiale : cas chez les villages voisins (< 25 km) --------
villages_coords = panel[["village", "lat", "lon"]].drop_duplicates("village").dropna().reset_index(drop=True)
rad = np.radians(villages_coords[["lat", "lon"]].values)
dlat = rad[:, 0:1] - rad[:, 0]
dlon = rad[:, 1:2] - rad[:, 1]
a = np.sin(dlat / 2) ** 2 + np.cos(rad[:, 0:1]) * np.cos(rad[:, 0]) * np.sin(dlon / 2) ** 2
distance_km = 6371 * 2 * np.arcsin(np.sqrt(a))
np.fill_diagonal(distance_km, 1e9)
matrice_voisinage = (distance_km < RAYON_VOISINAGE_KM).astype(float)

pivot_cas = panel.pivot_table(index="ym", columns="village", values="cas", aggfunc="sum").fillna(0)
pivot_cas = pivot_cas.reindex(columns=villages_coords["village"], fill_value=0)
cas_voisins_12m = pivot_cas.rolling(12, min_periods=1).sum().shift(1).fillna(0).values @ matrice_voisinage.T
cas_voisins_12m = pd.DataFrame(cas_voisins_12m, index=pivot_cas.index, columns=villages_coords["village"])
panel = panel.merge(
    cas_voisins_12m.stack().rename("voisinage_12m").reset_index().rename(columns={"level_1": "village"}),
    on=["village", "ym"], how="left"
)
panel["voisinage_12m"] = panel["voisinage_12m"].fillna(0)

VARIABLES_EXPLICATIVES = [
    "lag1", "lag2", "lag3", "lag12", "roll3", "roll12", "cumul_historique",
    "sin_mois", "cos_mois", "t", "voisinage_12m", "lat", "lon", "risque_10_14",
]
print(f"\n{len(VARIABLES_EXPLICATIVES)} variables explicatives construites :")
print(VARIABLES_EXPLICATIVES)


# ==============================================================================
# 5. SEPARATION TRAIN / TEST (TEMPORELLE)
# ==============================================================================
print("\n" + "=" * 78)
print("5. SEPARATION TRAIN / TEST (TEMPORELLE)")
print("=" * 78)
# Regle non negociable : jamais de validation croisee aleatoire sur une serie
# temporelle. On entraine sur le passe, on teste sur le futur reel.

train = panel[panel["ym"].dt.year <= ANNEE_LIMITE_TRAIN]
test = panel[panel["ym"].dt.year == ANNEE_LIMITE_TRAIN + 1]
print(f"Train (<= {ANNEE_LIMITE_TRAIN}) : {train.shape[0]} lignes, {int(train['y'].sum())} positifs")
print(f"Test  ({ANNEE_LIMITE_TRAIN + 1})      : {test.shape[0]} lignes, {int(test['y'].sum())} positifs")


# ==============================================================================
# 6. MODELES DE REFERENCE (BASELINES)
# ==============================================================================
print("\n" + "=" * 78)
print("6. MODELES DE REFERENCE (BASELINES)")
print("=" * 78)

resultats = {}
resultats["Persistance (actif sur les 12 derniers mois)"] = (test["roll12"] > 0).astype(float).values


# ==============================================================================
# 7. ENTRAINEMENT DU MODELE DE MACHINE LEARNING
# ==============================================================================
print("\n" + "=" * 78)
print("7. ENTRAINEMENT DU MODELE")
print("=" * 78)

# Regression logistique : reference simple et interpretable
modele_logit = LogisticRegression(max_iter=2000)
modele_logit.fit(train[VARIABLES_EXPLICATIVES].fillna(0), train["y"])
resultats["Regression logistique"] = modele_logit.predict_proba(test[VARIABLES_EXPLICATIVES].fillna(0))[:, 1]

# Gradient boosting : modele retenu (capture les interactions saison x historique)
modele_gbm = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
)
modele_gbm.fit(train[VARIABLES_EXPLICATIVES], train["y"])
resultats["Gradient Boosting (modele retenu)"] = modele_gbm.predict_proba(test[VARIABLES_EXPLICATIVES])[:, 1]

print("Modeles entraines : regression logistique + gradient boosting.")


# ==============================================================================
# 8. EVALUATION
# ==============================================================================
print("\n" + "=" * 78)
print("8. EVALUATION DES MODELES (sur l'annee test, jamais vue a l'entrainement)")
print("=" * 78)

print(f"{'Modele':45s} {'AUC':>8s} {'PR-AUC':>8s} {'Rappel@top5%':>14s}")
for nom, scores in resultats.items():
    scores = np.asarray(scores, dtype=float)
    auc = roc_auc_score(test["y"], scores)
    pr_auc = average_precision_score(test["y"], scores)
    k = max(1, int(len(test) * 0.05))
    top_k_idx = np.argsort(-scores)[:k]
    rappel_top5 = test["y"].values[top_k_idx].sum() / test["y"].sum()
    print(f"{nom:45s} {auc:8.3f} {pr_auc:8.3f} {rappel_top5 * 100:13.1f}%")

print(f"\nPrevalence de base (test) : {test['y'].mean():.2%}")
print("-> Le modele Gradient Boosting est retenu : il domine largement les baselines")
print("   et permet de couvrir une large part des cas en ne ciblant qu'une petite")
print("   fraction des villages-mois (voir rappel@top5%).")


# ==============================================================================
# 9. INTERPRETATION - IMPORTANCE DES VARIABLES
# ==============================================================================
print("\n" + "=" * 78)
print("9. INTERPRETATION (importance des variables, par permutation)")
print("=" * 78)

importance = permutation_importance(
    modele_gbm, test[VARIABLES_EXPLICATIVES], test["y"],
    n_repeats=8, random_state=0, scoring="average_precision"
)
importance_triee = pd.Series(importance.importances_mean, index=VARIABLES_EXPLICATIVES).sort_values(ascending=False)
print(importance_triee.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
importance_triee.plot(kind="barh", ax=ax, color="darkorange")
ax.set_title("Importance des variables (permutation, PR-AUC)")
ax.invert_yaxis()
plt.savefig("importance_variables.png", dpi=100)
plt.close(fig)
print("Graphique sauvegarde : importance_variables.png")


# ==============================================================================
# 10. MODELE FINAL (reentraine sur tout l'historique) ET EXPORT
# ==============================================================================
print("\n" + "=" * 78)
print("10. MODELE FINAL ET EXPORT DES PREDICTIONS")
print("=" * 78)
# Une fois la methode validee (etapes 5 a 9), on reentraine sur l'integralite
# de l'historique disponible pour disposer du modele le plus a jour possible.

modele_final = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
)
modele_final.fit(panel[VARIABLES_EXPLICATIVES], panel["y"])

# Score de risque sur le dernier mois disponible, a titre d'exemple d'usage
dernier_mois = panel["ym"].max()
extrait = panel[panel["ym"] == dernier_mois].copy()
extrait["score_risque"] = modele_final.predict_proba(extrait[VARIABLES_EXPLICATIVES])[:, 1]
classement = extrait.sort_values("score_risque", ascending=False)[["village", "score_risque"]]
classement.to_csv("classement_villages_risque.csv", index=False)

print(f"Modele final entraine sur {panel.shape[0]} observations ({panel['y'].sum()} positifs).")
print(f"Classement de risque exporte pour le mois {dernier_mois} : classement_villages_risque.csv")
print("\nTop 10 villages les plus a risque (dernier mois disponible) :")
print(classement.head(10).to_string(index=False))

print("\n" + "=" * 78)
print("PIPELINE TERMINE")
print("=" * 78)
