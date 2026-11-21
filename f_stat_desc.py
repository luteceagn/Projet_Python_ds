import textwrap
import geopandas as gpd
from matplotlib import patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

URL_GEO_DEPS_DEFAULT = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"


def tracer_top_departements_meteo(
    df,
    indicateurs=("RR", "NBJFF16"),
    top_n=10,
    date_min=201801,
    figsize=(14, 5.5),
    meta_custom=None,
):
    """Trace côte à côte le Top N des départements pour deux indicateurs météo.

    Paramètres :
    - df : DataFrame contenant DATE, departement, les colonnes d'indicateurs et optionnellement NOM_DEP.
    - indicateurs : tuple/liste de 2 indicateurs à comparer (ex. ("RR", "NBJFF16")).
    - top_n : nombre de départements classés par panneau (défaut : 10).
    """
    if len(indicateurs) != 2:
        raise ValueError(
            "Le paramètre 'indicateurs' doit contenir exactement 2 colonnes."
        )

    for ind in indicateurs:
        if ind not in df.columns:
            raise ValueError(
                f"La colonne '{ind}' est absente du DataFrame fourni."
            )

    meta_indicateurs = {
        "RR": (
            "cumul pluie moyen (RR)",
            "Précipitations mensuelles moyennes (mm)",
            "#2b8cbe",
        ),
        "NBJRR10": (
            "jours de forte pluie >= 10 mm (NBJRR10)",
            "Part des jours avec forte pluie (%)",
            "#1b9e77",
        ),
        "NBJFF16": (
            "fréquence vent fort (NBJFF16)",
            "Part des jours avec vent fort (%)",
            "#7570b3",
        ),
        "NBJGELEE": (
            "fréquence jours de gel (NBJGELEE)",
            "Part des jours de gel (%)",
            "#3182bd",
        ),
        "NBJTX30": (
            "fréquence forte chaleur > 30°C (NBJTX30)",
            "Part des jours de forte chaleur (%)",
            "#d95f02",
        ),
        "NBJORAG": (
            "fréquence jours d'orage (NBJORAG)",
            "Part des jours d'orage (%)",
            "#e7298a",
        ),
        "NBJBROU": (
            "fréquence jours de brouillard (NBJBROU)",
            "Part des jours de brouillard (%)",
            "#636363",
        ),
        "TN": (
            "température minimale moyenne (TN)",
            "Température minimale mensuelle moyenne (°C)",
            "#2c7bb6",
        ),
        "TX": (
            "température maximale moyenne (TX)",
            "Température maximale mensuelle moyenne (°C)",
            "#d7191c",
        ),
        "anomalie_tn": (
            "anomalie température minimale (anomalie_tn)",
            "Écart à la normale de TN (°C)",
            "#4575b4",
        ),
        "anomalie_tx": (
            "anomalie température maximale (anomalie_tx)",
            "Écart à la normale de TX (°C)",
            "#d73027",
        ),
        "anomalie_rr": (
            "anomalie précipitations (anomalie_rr)",
            "Écart absolu à la normale (mm)",
            "#3182bd",
        ),
        "anomalie_rr_pct": (
            "anomalie relative précipitations (anomalie_rr_pct)",
            "Écart relatif à la normale (%)",
            "#6baed6",
        ),
    }
    if meta_custom:
        meta_indicateurs.update(meta_custom)

    # 1. Filtrage temporel et construction autonome de dep_label
    df_filtre = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_filtre["DATE"]):
        codes_ym = df_filtre["DATE"].dt.strftime("%Y%m").astype(int)
    else:
        codes_ym = (
            df_filtre["DATE"].astype(str).str.slice(0, 6).astype(int)
        )

    df_filtre = df_filtre[codes_ym >= int(date_min)].copy()
    dep_code = df_filtre["departement"].astype(str).str.zfill(2)

    if "NOM_DEP" in df_filtre.columns:
        df_filtre["dep_label"] = df_filtre["NOM_DEP"].fillna(dep_code)
    else:
        df_filtre["dep_label"] = dep_code

    # 2. Agrégation moyenne par département sur les indicateurs demandés
    df_spatial = df_filtre.groupby("dep_label", as_index=False)[
        list(indicateurs)
    ].mean()

    # 3. Tracé côte à côte
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    couleurs_secours = ["#2b8cbe", "#7570b3"]

    for idx, (ax, ind) in enumerate(zip(axes, indicateurs)):
        if ind in meta_indicateurs:
            libelle_court, xlabel, couleur = meta_indicateurs[ind]
        else:
            libelle_court = ind
            xlabel = ind
            couleur = couleurs_secours[idx]

        top_df = df_spatial.nlargest(top_n, ind).sort_values(
            ind, ascending=True
        )
        ax.barh(
            top_df["dep_label"], top_df[ind], color=couleur, edgecolor="none"
        )
        ax.set_title(
            f"Prisme spatial — Top {top_n} départements : {libelle_court}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Département")

    plt.tight_layout()
    plt.show()

def tracer_heatmap_meteo(
    df,
    deps,
    indicateur,
    date_min=201801,
    figsize=(15, 5.5),
    xticklabels=6,
    meta_custom=None,
):
    """Trace une heatmap Temps x Espace pour une liste de départements et un indicateur donné.

    Paramètres :
    - df : DataFrame contenant DATE, departement, l'indicateur et optionnellement NOM_DEP.
    - deps : liste/séquence de codes départements (ex. ["75", "11", "69", ...]).
    - indicateur : nom de la colonne à afficher (ex. "anomalie_tn", "RR", "NBJGELEE"...).
    """
    if indicateur not in df.columns:
        raise ValueError(
            f"La colonne '{indicateur}' est absente du DataFrame fourni."
        )

    meta_indicateurs = {
        "RR": (
            "Cumul mensuel de précipitations (RR)",
            "Précipitations (mm)",
            "Blues",
            None,
        ),
        "NBJRR10": (
            "Jours de forte pluie >= 10 mm (NBJRR10)",
            "Part des jours du mois (%)",
            "BuGn",
            None,
        ),
        "NBJFF16": (
            "Fréquence des jours de vent fort (NBJFF16)",
            "Part des jours du mois (%)",
            "Purples",
            None,
        ),
        "NBJGELEE": (
            "Fréquence des jours de gel (NBJGELEE)",
            "Part des jours du mois (%)",
            "Blues",
            None,
        ),
        "NBJTX30": (
            "Fréquence des jours de forte chaleur > 30°C (NBJTX30)",
            "Part des jours du mois (%)",
            "OrRd",
            None,
        ),
        "NBJORAG": (
            "Fréquence des jours d'orage (NBJORAG)",
            "Part des jours du mois (%)",
            "PuRd",
            None,
        ),
        "NBJBROU": (
            "Fréquence des jours de brouillard (NBJBROU)",
            "Part des jours du mois (%)",
            "Greys",
            None,
        ),
        "TN": (
            "Température minimale mensuelle (TN)",
            "Température minimale (°C)",
            "coolwarm",
            None,
        ),
        "TX": (
            "Température maximale mensuelle (TX)",
            "Température maximale (°C)",
            "coolwarm",
            None,
        ),
        "anomalie_tn": (
            "Anomalies de température minimales (anomalie_tn)",
            "Anomalie TN (°C)",
            "coolwarm",
            0,
        ),
        "anomalie_tx": (
            "Anomalies de température maximales (anomalie_tx)",
            "Anomalie TX (°C)",
            "coolwarm",
            0,
        ),
        "anomalie_rr": (
            "Anomalies de précipitations (anomalie_rr)",
            "Écart à la normale (mm)",
            "coolwarm",
            0,
        ),
        "anomalie_rr_pct": (
            "Anomalies relatives de précipitations (anomalie_rr_pct)",
            "Écart à la normale (%)",
            "coolwarm",
            0,
        ),
    }
    if meta_custom:
        meta_indicateurs.update(meta_custom)

    # 1. Récupération des métadonnées (titre, légende cbar, palette, centre)
    if indicateur in meta_indicateurs:
        sujet_titre, label_cbar, cmap, center = meta_indicateurs[indicateur]
    else:
        sujet_titre = f"Évolution de {indicateur}"
        label_cbar = indicateur
        cmap = "coolwarm" if indicateur.startswith("anomalie_") else "YlGnBu"
        center = 0 if indicateur.startswith("anomalie_") else None

    # 2. Préparation temporelle et géographique autonome
    deps_cles = [str(d).zfill(2) for d in deps]
    df_hm = df.copy()

    if pd.api.types.is_datetime64_any_dtype(df_hm["DATE"]):
        df_hm["dt"] = df_hm["DATE"]
    else:
        df_hm["dt"] = pd.to_datetime(
            df_hm["DATE"].astype(str).str.slice(0, 6), format="%Y%m"
        )

    df_hm = df_hm[df_hm["dt"].dt.strftime("%Y%m").astype(int) >= int(date_min)]
    df_hm["dep_str"] = df_hm["departement"].astype(str).str.zfill(2)
    df_hm = df_hm[df_hm["dep_str"].isin(deps_cles)].copy()

    if "NOM_DEP" in df_hm.columns:
        df_hm["dep_label"] = df_hm["NOM_DEP"].fillna(df_hm["dep_str"])
    else:
        df_hm["dep_label"] = df_hm["dep_str"]

    df_hm["annee_mois"] = df_hm["dt"].dt.strftime("%Y-%m")

    # Conserve l'ordre exact de deps_cles traduit en dep_label
    ordre_labels = (
        df_hm.drop_duplicates("dep_str")
        .set_index("dep_str")
        .reindex(deps_cles)["dep_label"]
        .dropna()
        .tolist()
    )

    # 3. Pivot (agrégation par moyenne au cas où plusieurs stations existent par département/mois)
    pivot_vals = (
        df_hm.groupby(["dep_label", "annee_mois"], as_index=False)[indicateur]
        .mean()
        .pivot(index="dep_label", columns="annee_mois", values=indicateur)
        .reindex(ordre_labels)
    )

    # 4. Tracé de la heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap_kwargs = {
        "cmap": cmap,
        "cbar_kws": {"label": label_cbar},
        "xticklabels": xticklabels,
        "ax": ax,
    }
    if center is not None:
        heatmap_kwargs["center"] = center

    sns.heatmap(pivot_vals, **heatmap_kwargs)

    nb_deps = len(ordre_labels)
    ax.set_title(
        f"Prisme Temps × Espace — {sujet_titre} sur {nb_deps} départements clés",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Période (Année-Mois)")
    ax.set_ylabel("Département")

    plt.tight_layout()
    plt.show()

def tracer_carte_meteo(
    df,
    indicateur,
    gdf_fond=None,
    date_min=201801,
    figsize=(10, 10),
    labels=False,
    meta_custom=None,
    url_geo=URL_GEO_DEPS_DEFAULT,
):
    """Trace une carte choroplèthe des départements métropolitains pour l'indicateur choisi.

    Paramètres :
    - labels : False (aucune étiquette), "nom" ou True (colonne 'nom' du GeoJSON), ou "code" (numéro).
    """
    if indicateur not in df.columns:
        raise ValueError(
            f"La colonne '{indicateur}' est absente du DataFrame fourni."
        )

    meta_indicateurs = {
        "RR": (
            "Cumul mensuel moyen de précipitations par département",
            "Précipitations moyennes (mm)",
            "Blues",
        ),
        "NBJRR10": (
            "Fréquence mensuelle des jours de forte pluie (>= 10 mm)",
            "Part des jours du mois (%)",
            "BuGn",
        ),
        "NBJFF16": (
            "Fréquence mensuelle des jours de vent fort",
            "Part des jours du mois (%)",
            "Purples",
        ),
        "NBJGELEE": (
            "Fréquence mensuelle des jours de gel",
            "Part des jours du mois (%)",
            "Blues",
        ),
        "NBJTX30": (
            "Fréquence mensuelle des jours de forte chaleur (> 30°C)",
            "Part des jours du mois (%)",
            "OrRd",
        ),
        "NBJORAG": (
            "Fréquence mensuelle des jours d'orage",
            "Part des jours du mois (%)",
            "PuRd",
        ),
        "NBJBROU": (
            "Fréquence mensuelle des jours de brouillard",
            "Part des jours du mois (%)",
            "Greys",
        ),
        "TN": (
            "Température minimale mensuelle moyenne",
            "Température minimale (°C)",
            "coolwarm",
        ),
        "TX": (
            "Température maximale mensuelle moyenne",
            "Température maximale (°C)",
            "coolwarm",
        ),
        "anomalie_tn": (
            "Anomalie de température minimale (écart à la normale)",
            "Écart à la normale (°C)",
            "coolwarm",
        ),
        "anomalie_tx": (
            "Anomalie de température maximale (écart à la normale)",
            "Écart à la normale (°C)",
            "coolwarm",
        ),
        "anomalie_rr": (
            "Anomalie de précipitations (écart absolu à la normale)",
            "Écart à la normale (mm)",
            "coolwarm",
        ),
        "anomalie_rr_pct": (
            "Anomalie relative de précipitations (écart en % à la normale)",
            "Écart à la normale (%)",
            "coolwarm",
        ),
    }
    if meta_custom:
        meta_indicateurs.update(meta_custom)

    # 1. Récupération des titres et de la palette
    if indicateur in meta_indicateurs:
        titre_base, titre_cbar, cmap = meta_indicateurs[indicateur]
    else:
        titre_base = f"Répartition départementale de {indicateur}"
        titre_cbar = indicateur
        cmap = "Blues"

    # 2. Chargement du fond de carte (fourni, mis en cache sur la fonction, ou téléchargé)
    if gdf_fond is None:
        if not hasattr(tracer_carte_meteo, "_gdf_cache"):
            gdf_temp = gpd.read_file(url_geo)
            gdf_temp["code_jointure"] = gdf_temp["code"].astype(str).str.zfill(2)
            tracer_carte_meteo._gdf_cache = gdf_temp
        gdf_base = tracer_carte_meteo._gdf_cache.copy()
    else:
        gdf_base = gdf_fond.copy()
        col_code_geo = "code" if "code" in gdf_base.columns else "code_dep"
        gdf_base["code_jointure"] = gdf_base[col_code_geo].astype(str).str.zfill(2)

    # 3. Agrégation moyenne par département sur la période choisie
    df_vals = (
        df[df["DATE"] >= date_min]
        .copy()
        .assign(code_dep=lambda d: d["departement"].astype(str).str.zfill(2))
        .groupby("code_dep", as_index=False)[indicateur]
        .mean()
    )

    # 4. Jointure avec le fond de carte
    gdf_merged = gdf_base.merge(
        df_vals, left_on="code_jointure", right_on="code_dep", how="left"
    )

    # 5. Gestion de l'échelle : symétrique autour de 0 si anomalie, sinon min/max naturel
    plot_kwargs = {
        "column": indicateur,
        "cmap": cmap,
        "linewidth": 0.6,
        "edgecolor": "white",
        "missing_kwds": {
            "color": "lightgrey",
            "label": "Donnée absente / hors périmètre",
        },
        "legend": True,
        "legend_kwds": {"label": titre_cbar, "shrink": 0.75},
    }

    if indicateur.startswith("anomalie_"):
        bornes = gdf_merged[indicateur].dropna()
        if not bornes.empty:
            max_abs = float(np.max(np.abs(bornes)))
            plot_kwargs["vmin"] = -max_abs
            plot_kwargs["vmax"] = max_abs

    # 6. Tracé de la carte
    fig, ax = plt.subplots(figsize=figsize)
    gdf_merged.plot(ax=ax, **plot_kwargs)

    # 7. Ajout optionnel des étiquettes ("nom" du GeoJSON ou "code")
    if labels:
        deps_idf = {"75", "77", "78", "91", "92", "93", "94", "95"}
        for _, row in gdf_merged.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            pt = row.geometry.representative_point()
            code_dep = str(row["code_jointure"])

            if labels == "code" or code_dep in deps_idf:
                texte = code_dep
                taille = 6.2 if code_dep in deps_idf else 7.3
                interligne = 1.0
            elif (
                labels in ("nom", True)
                and "nom" in gdf_merged.columns
                and pd.notna(row["nom"])
            ):
                texte = textwrap.fill(str(row["nom"]), width=11)
                taille = 6.2
                interligne = 0.9
            else:
                texte = code_dep
                taille = 7.0
                interligne = 1.0

            txt = ax.text(
                pt.x,
                pt.y,
                texte,
                ha="center",
                va="center",
                linespacing=interligne,
                fontsize=taille,
                fontweight="bold",
                color="#111111",
            )
            txt.set_path_effects([
                patheffects.withStroke(
                    linewidth=2, foreground="white", alpha=0.85
                )
            ])

    annee_debut = str(date_min)[:4]
    ax.set_title(
        f"{titre_base} ({annee_debut}–2026)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.show()
