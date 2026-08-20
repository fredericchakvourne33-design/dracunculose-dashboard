"""
==============================================================================
TABLEAU DE BORD DE SURVEILLANCE - VER DE GUINEE (TCHAD)
Application Streamlit a executer en local.

Lancement :
    streamlit run app_dashboard.py

Fonctions :
  - Historique : infections par province et par mois (2022-2025)
  - Prevision 2026 par province (modele national reconcilie avec des modeles
    province, methode identique aux notebooks livres precedemment)
  - Districts a risque en 2026, filtrable par province (modele de
    classification entraine sur l'historique complet, projection recursive
    mois par mois pour 2026)
==============================================================================
"""

import unicodedata
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

st.set_page_config(page_title="Surveillance Ver de Guinée - Tchad", page_icon="🐍", layout="wide")

FENETRE_INCUBATION = (10, 14)   # periode d'incubation confirmee (mois)
MOIS_2026 = pd.period_range("2026-01", "2026-12", freq="M")
SEUIL_CAS_PROVINCE = 100        # en-dessous, pas assez d'historique pour une tendance fiable
FENETRE_MODELE_MENSUEL = 24     # fenetre glissante retenue par backtest (voir notebooks)


# ==========================================================================
# NETTOYAGE
# ==========================================================================
def normaliser_texte(valeur):
    if pd.isna(valeur):
        return np.nan
    valeur = str(valeur).strip()
    valeur = unicodedata.normalize("NFKD", valeur).encode("ascii", "ignore").decode()
    return re.sub(r"[\s_]+", " ", valeur).strip().upper()


def parser_date(valeur):
    if pd.isna(valeur):
        return pd.NaT
    valeur = str(valeur).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return pd.to_datetime(valeur, format=fmt)
        except ValueError:
            continue
    return pd.NaT


@st.cache_data(show_spinner="Chargement et nettoyage des données…")
def charger_et_nettoyer(fichier):
    df = pd.read_csv(fichier, sep=";", encoding="utf-8-sig")
    df["date_emergence"] = df["Date d'emergence"].apply(parser_date)
    df["lat"] = pd.to_numeric(df["Latitude"].astype(str).str.replace(",", "."), errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude"].astype(str).str.replace(",", "."), errors="coerce")
    df["province_n"] = df["Province"].apply(normaliser_texte)
    df["district_n"] = df["District"].apply(normaliser_texte)
    df["isolee_n"] = df["Infection Isolee?"].apply(normaliser_texte)
    df["contam_n"] = df["Contamination des Sources d'eau"].apply(normaliser_texte)
    df["attache_n"] = df["Animal Attache"].apply(normaliser_texte)
    df = df.dropna(subset=["date_emergence"]).copy()
    df["ym"] = df["date_emergence"].dt.to_period("M")
    df["risque_transmission"] = (
        (df["isolee_n"] == "NON") & ((df["contam_n"] == "OUI") | (df["attache_n"] == "NON"))
    ).astype(int)
    mois_hist_local = pd.period_range(df["ym"].min(), "2025-12", freq="M")
    return df, mois_hist_local


# ==========================================================================
# PREVISION 2026 - VOLUME (national reconcilie avec les provinces)
# Meme methode que les notebooks Prevision_Infections_2026 et
# Prevision_Provinces_2026 livres precedemment.
# ==========================================================================
@st.cache_data(show_spinner="Entraînement des modèles de prévision (volume)…")
def previsions_volume_2026(df, _mois_hist):
    # NB : l'argument est prefixe par un underscore (_mois_hist) car un
    # PeriodIndex n'est pas hachable par le systeme de cache de Streamlit ;
    # ce prefixe indique a Streamlit de ne pas essayer de le hacher.
    mois_hist = _mois_hist
    # --- national ---
    s = df.groupby("ym").size().reindex(mois_hist, fill_value=0).rename("cas").to_frame()
    s["mois"] = s.index.month
    s["t"] = np.arange(len(s))
    s["sin"] = np.sin(2 * np.pi * s.mois / 12)
    s["cos"] = np.cos(2 * np.pi * s.mois / 12)
    feats = ["t", "sin", "cos"]
    train = s.iloc[-FENETRE_MODELE_MENSUEL:]
    modele_national = PoissonRegressor(alpha=0.5, max_iter=2000).fit(train[feats], train["cas"])

    fut_nat = pd.DataFrame(index=MOIS_2026)
    fut_nat["mois"] = fut_nat.index.month
    fut_nat["t"] = np.arange(len(s), len(s) + 12)
    fut_nat["sin"] = np.sin(2 * np.pi * fut_nat.mois / 12)
    fut_nat["cos"] = np.cos(2 * np.pi * fut_nat.mois / 12)
    fut_nat["total_national"] = modele_national.predict(fut_nat[feats])

    # --- par province ---
    provinces = sorted(df.province_n.dropna().unique())
    totaux_provinces = df.groupby("province_n").size()
    previsions_brutes = []
    for prov in provinces:
        sous = df[df.province_n == prov]
        sp = sous.groupby("ym").size().reindex(mois_hist, fill_value=0).rename("cas").to_frame()
        sp["mois"] = sp.index.month
        sp["t"] = np.arange(len(sp))
        sp["sin"] = np.sin(2 * np.pi * sp.mois / 12)
        sp["cos"] = np.cos(2 * np.pi * sp.mois / 12)

        fut = pd.DataFrame(index=MOIS_2026)
        fut["mois"] = fut.index.month
        fut["sin"] = np.sin(2 * np.pi * fut.mois / 12)
        fut["cos"] = np.cos(2 * np.pi * fut.mois / 12)

        if totaux_provinces.get(prov, 0) >= SEUIL_CAS_PROVINCE:
            trp = sp.iloc[-FENETRE_MODELE_MENSUEL:]
            mp = PoissonRegressor(alpha=0.5, max_iter=2000).fit(trp[feats], trp["cas"])
            fut["t"] = np.arange(len(sp), len(sp) + 12)
            fut["pred_brute"] = mp.predict(fut[feats])
        else:
            moyenne = sp.groupby("mois")["cas"].mean()
            fut["pred_brute"] = fut["mois"].map(moyenne).fillna(0)

        fut["province"] = prov
        previsions_brutes.append(fut.reset_index().rename(columns={"index": "ym"}))

    previsions_brutes = pd.concat(previsions_brutes, ignore_index=True)

    # --- reconciliation top-down : le national fait foi pour le volume,
    # les provinces ne fournissent que la cle de repartition mensuelle ---
    nat = fut_nat.reset_index().rename(columns={"index": "ym"})[["ym", "total_national"]]
    fus = previsions_brutes.merge(nat, on="ym", how="left")
    total_mensuel = fus.groupby("ym")["pred_brute"].transform("sum")
    fus["part"] = fus["pred_brute"] / total_mensuel.replace(0, np.nan)
    fus["prediction"] = (fus["part"] * fus["total_national"]).fillna(0)

    return fus[["province", "ym", "prediction"]], nat


# ==========================================================================
# PREVISION 2026 - DISTRICTS A RISQUE (classification, projection recursive)
# ==========================================================================
FEATURES_DISTRICT = ["lag1", "lag2", "lag3", "lag12", "roll3", "roll12", "cum",
                      "sin", "cos", "t", "lat", "lon", "risque_10_14"]


def construire_panel_district(df, mois_hist):
    districts = sorted(df.district_n.dropna().unique())
    mapping_province = (
        df.dropna(subset=["district_n"]).groupby("district_n")["province_n"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    idx = pd.MultiIndex.from_product([districts, mois_hist], names=["district", "ym"])
    panel = pd.DataFrame(index=idx).reset_index()

    cas = df.groupby(["district_n", "ym"]).size().rename("cas")
    risq = df.groupby(["district_n", "ym"])["risque_transmission"].sum().rename("risque")
    panel = panel.merge(cas, left_on=["district", "ym"], right_index=True, how="left").fillna({"cas": 0})
    panel = panel.merge(risq, left_on=["district", "ym"], right_index=True, how="left").fillna({"risque": 0})
    panel["y"] = (panel.cas > 0).astype(int)
    panel["province"] = panel["district"].map(mapping_province)

    coords = df.dropna(subset=["lat", "lon"]).groupby("district_n")[["lat", "lon"]].median()
    panel = panel.merge(coords, left_on="district", right_index=True, how="left")

    g = panel.groupby("district")
    for retard in [1, 2, 3, 12]:
        panel[f"lag{retard}"] = g["cas"].shift(retard)
    panel["roll3"] = g["cas"].shift(1).transform(lambda s: s.rolling(3, min_periods=1).sum())
    panel["roll12"] = g["cas"].shift(1).transform(lambda s: s.rolling(12, min_periods=1).sum())
    panel["cum"] = g["cas"].cumsum() - panel["cas"]
    panel["risque_10_14"] = g["risque"].transform(
        lambda s: s.shift(FENETRE_INCUBATION[0]).rolling(
            FENETRE_INCUBATION[1] - FENETRE_INCUBATION[0] + 1, min_periods=1
        ).sum()
    )
    panel["mois"] = panel.ym.dt.month
    panel["sin"] = np.sin(2 * np.pi * panel.mois / 12)
    panel["cos"] = np.cos(2 * np.pi * panel.mois / 12)
    panel["t"] = (panel.ym.dt.year - panel.ym.dt.year.min()) * 12 + panel.mois
    return panel.sort_values(["district", "ym"]).reset_index(drop=True)


@st.cache_data(show_spinner="Entraînement et validation du modèle district…")
def entrainer_et_valider_district(df, _mois_hist):
    # meme raison que ci-dessus : underscore pour eviter le hachage du PeriodIndex
    panel = construire_panel_district(df, _mois_hist)

    # validation temporelle : entraine <= 2024, teste sur 2025 (jamais l'inverse)
    train = panel[panel.ym.dt.year <= 2024]
    test = panel[panel.ym.dt.year == 2025]
    metriques = None
    if len(test) and test["y"].sum() > 0:
        m_valid = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.07, max_leaf_nodes=12, l2_regularization=1., random_state=0
        ).fit(train[FEATURES_DISTRICT], train.y)
        proba = m_valid.predict_proba(test[FEATURES_DISTRICT])[:, 1]
        metriques = {
            "AUC": roc_auc_score(test.y, proba),
            "PR_AUC": average_precision_score(test.y, proba),
            "n_test": len(test),
            "positifs_test": int(test.y.sum()),
        }

    # modele final : reentraine sur tout l'historique disponible
    modele_final = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=.07, max_leaf_nodes=12, l2_regularization=1., random_state=0
    ).fit(panel[FEATURES_DISTRICT], panel.y)

    return panel, modele_final, metriques


@st.cache_data(show_spinner="Projection des districts à risque pour 2026…")
def prevoir_districts_2026(_panel, _modele):
    panel, modele = _panel, _modele
    coords = panel[["district", "lat", "lon", "province"]].drop_duplicates("district").set_index("district")
    cas_series = panel.pivot_table(index="ym", columns="district", values="cas", aggfunc="sum").fillna(0)
    risque_series = panel.pivot_table(index="ym", columns="district", values="risque", aggfunc="sum").fillna(0)
    annee_min = panel.ym.dt.year.min()
    resultats = []
    for mois_cible in MOIS_2026:
        hist = cas_series[cas_series.index < mois_cible]
        lag1 = hist.iloc[-1] if len(hist) >= 1 else pd.Series(0, index=cas_series.columns)
        lag2 = hist.iloc[-2] if len(hist) >= 2 else pd.Series(0, index=cas_series.columns)
        lag3 = hist.iloc[-3] if len(hist) >= 3 else pd.Series(0, index=cas_series.columns)
        lag12 = hist.iloc[-12] if len(hist) >= 12 else pd.Series(0, index=cas_series.columns)
        roll3 = hist.iloc[-3:].sum() if len(hist) >= 1 else pd.Series(0, index=cas_series.columns)
        roll12 = hist.iloc[-12:].sum() if len(hist) >= 1 else pd.Series(0, index=cas_series.columns)
        cum = hist.sum()
        fen = risque_series[
            (risque_series.index >= mois_cible - FENETRE_INCUBATION[1])
            & (risque_series.index <= mois_cible - FENETRE_INCUBATION[0])
        ]
        risque_10_14 = fen.sum() if len(fen) else pd.Series(0, index=cas_series.columns)

        feat = pd.DataFrame({"district": cas_series.columns})
        feat["lag1"] = feat.district.map(lag1); feat["lag2"] = feat.district.map(lag2)
        feat["lag3"] = feat.district.map(lag3); feat["lag12"] = feat.district.map(lag12)
        feat["roll3"] = feat.district.map(roll3); feat["roll12"] = feat.district.map(roll12)
        feat["cum"] = feat.district.map(cum); feat["risque_10_14"] = feat.district.map(risque_10_14)
        feat["lat"] = feat.district.map(coords["lat"]); feat["lon"] = feat.district.map(coords["lon"])
        feat["mois"] = mois_cible.month
        feat["sin"] = np.sin(2 * np.pi * feat.mois / 12); feat["cos"] = np.cos(2 * np.pi * feat.mois / 12)
        feat["t"] = (mois_cible.year - annee_min) * 12 + mois_cible.month
        feat = feat.dropna(subset=["lat", "lon"])

        proba = modele.predict_proba(feat[FEATURES_DISTRICT])[:, 1]
        feat["proba"] = proba
        feat["ym"] = mois_cible
        feat["province"] = feat.district.map(coords["province"])
        resultats.append(feat[["ym", "district", "province", "proba"]])

        pseudo = pd.Series(0.0, index=cas_series.columns)
        pseudo.update(feat.set_index("district")["proba"])
        cas_series.loc[mois_cible] = pseudo
        risque_series.loc[mois_cible] = 0.0

    return pd.concat(resultats, ignore_index=True)


# ==========================================================================
# INTERFACE
# ==========================================================================
st.title("🐍 Surveillance du ver de Guinée — Tableau de bord")
st.caption(
    "Système intelligent de surveillance et d'aide à la décision — infections animales, Tchad. "
    "Analyse basée uniquement sur `Infections_Animales.csv`."
)

with st.sidebar:
    st.header("📂 Données")
    fichier_televerse = st.file_uploader("Charger le fichier CSV", type=["csv"])
    chemin_par_defaut = st.text_input("…ou chemin local vers le fichier", value="Infections_Animales.csv")

source = fichier_televerse if fichier_televerse is not None else chemin_par_defaut

try:
    df, mois_hist = charger_et_nettoyer(source)
except FileNotFoundError:
    st.error(f"Fichier introuvable : « {chemin_par_defaut} ». Téléversez le CSV via le panneau de gauche.")
    st.stop()
except Exception as erreur:
    st.error(f"Erreur lors de la lecture du fichier : {erreur}")
    st.stop()

st.sidebar.success(f"{len(df)} infections chargées ({df['ym'].min()} → {df['ym'].max()})")

st.sidebar.header("🔎 Filtre")
provinces_disponibles = sorted(df["province_n"].dropna().unique())
province_choisie = st.sidebar.selectbox("Province", options=["Toutes"] + provinces_disponibles)

# Calculs (mis en cache : ne se recalculent pas a chaque interaction)
previsions_provinces, previsions_nationales = previsions_volume_2026(df, mois_hist)
panel_district, modele_district, metriques_district = entrainer_et_valider_district(df, mois_hist)
previsions_districts = prevoir_districts_2026(panel_district, modele_district)

st.divider()

# ==========================================================================
# SECTION 1 - PREVISION 2026 : NOMBRE D'INFECTIONS
# ==========================================================================
if province_choisie == "Toutes":
    st.subheader("📈 Prévision 2026 — total national")
    serie_affichee = previsions_nationales.set_index("ym")["total_national"]
    titre_graph = "Nombre d'infections prévu par mois — toutes provinces (2026)"
else:
    st.subheader(f"📈 Prévision 2026 — {province_choisie}")
    serie_affichee = (
        previsions_provinces[previsions_provinces.province == province_choisie]
        .set_index("ym")["prediction"]
    )
    titre_graph = f"Nombre d'infections prévu par mois — {province_choisie} (2026)"

col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(serie_affichee.index.astype(str), serie_affichee.values, color="steelblue")
    ax.set_title(titre_graph)
    ax.set_ylabel("Cas prévus")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")
    st.pyplot(fig)
with col2:
    st.metric("Total prévu 2026", f"{serie_affichee.sum():.0f} cas")
    st.dataframe(serie_affichee.rename("Cas prévus").round(1), use_container_width=True)

st.caption(
    "Le total national est le modèle de référence, validé en test sur 2025 (MAE ≈ 4,5 cas/mois). "
    "La répartition par province est appliquée à ce total (méthode de réconciliation), pour rester "
    "cohérente d'une province à l'autre."
)

st.divider()

# ==========================================================================
# SECTION 2 - DISTRICTS A RISQUE EN 2026
# ==========================================================================
st.subheader(
    "🚨 Districts à risque en 2026" + (f" — {province_choisie}" if province_choisie != "Toutes" else "")
)

dist_filtre = previsions_districts.copy()
if province_choisie != "Toutes":
    dist_filtre = dist_filtre[dist_filtre.province == province_choisie]

if dist_filtre.empty:
    st.info("Aucun district avec coordonnées disponibles pour cette province.")
else:
    resume_district = (
        dist_filtre.groupby("district")["proba"]
        .agg(risque_moyen_2026="mean", risque_max_2026="max")
        .sort_values("risque_moyen_2026", ascending=False)
    )
    mois_pic = dist_filtre.loc[dist_filtre.groupby("district")["proba"].idxmax(), ["district", "ym", "proba"]]
    mois_pic = mois_pic.set_index("district")["ym"].astype(str).rename("mois_du_pic_de_risque")
    resume_district = resume_district.join(mois_pic)

    seuil_alerte = (
        resume_district["risque_moyen_2026"].quantile(0.75)
        if len(resume_district) > 3 else resume_district["risque_moyen_2026"].median()
    )
    resume_district["niveau"] = np.where(
        resume_district["risque_moyen_2026"] >= seuil_alerte, "Élevé", "Modéré/Faible"
    )

    a_risque = resume_district[resume_district["niveau"] == "Élevé"]
    if a_risque.empty:
        st.success("Aucun district ne se distingue nettement pour 2026 avec les données actuelles.")
    else:
        for district, ligne in a_risque.iterrows():
            st.error(
                f"🔴 **{district}** — score de risque moyen {ligne['risque_moyen_2026']:.3f}, "
                f"pic attendu en {ligne['mois_du_pic_de_risque']}."
            )

    with st.expander("Classement complet des districts (2026)"):
        st.dataframe(
            resume_district.round(4).reset_index().rename(columns={"district": "District"}),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "Score = probabilité moyenne, sur les 12 mois de 2026, qu'au moins un cas survienne dans le "
        "district (modèle de classification entraîné sur 2022-2025, variables : historique propre au "
        "district, saisonnalité, et signal de transmission différée à 10-14 mois). Le seuil « Élevé » "
        "correspond au quart des districts les plus exposés dans le périmètre affiché."
    )

if metriques_district:
    with st.expander("ℹ️ Fiabilité du modèle district (validation sur 2025)"):
        st.write(
            f"AUC = {metriques_district['AUC']:.3f} — PR-AUC = {metriques_district['PR_AUC']:.3f} "
            f"(mesuré sur {metriques_district['n_test']} district-mois de test, dont "
            f"{metriques_district['positifs_test']} réellement touchés — jamais vus à l'entraînement)."
        )

st.divider()
st.caption(
    "Prévisions construites uniquement à partir de l'historique disponible (aucune donnée externe). "
    "À utiliser comme aide à la priorisation, pas comme certitude — en particulier pour les districts "
    "n'ayant jamais été touchés auparavant, plus difficiles à anticiper."
)
