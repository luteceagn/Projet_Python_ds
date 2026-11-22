"""Fonctions statistiques descriptives pour la base SNCF-Meteo jointe."""

from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES_METEO = [
    "RR_depart", "TN_depart", "TX_depart", "TAMPLIM_depart",
    "NBJGELEE_depart", "NBJRR10_depart",
    "NBJORAG_depart", "NBJBROU_depart", "NBJFF16_depart", "NBJTX30_depart",
    "RR_arrivee", "TN_arrivee", "TX_arrivee", "TAMPLIM_arrivee",
    "NBJGELEE_arrivee", "NBJRR10_arrivee",
    "NBJORAG_arrivee", "NBJBROU_arrivee", "NBJFF16_arrivee", "NBJTX30_arrivee",
]


def charger_base(input_path="df_sncf_avec_meteo.csv"):
    """Charge la jointure et construit les indicateurs SNCF pondérés.

    Les circulations effectives valent les trains prévus moins les
    annulations. Les taux rapportent les retards aux circulations positives
    et les annulations aux trains prévus.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Base jointe introuvable : {path}. Executez Join_data.py."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["annee"] = df["date"].dt.year
    df["mois"] = df["date"].dt.month

    prevu = pd.to_numeric(df["nb_train_prevu"], errors="coerce")
    annule = pd.to_numeric(df["nb_annulation"], errors="coerce")
    retard = pd.to_numeric(df["nb_train_retard_arrivee"], errors="coerce")

    df["circulations_effectives"] = prevu - annule
    df["taux_retard_arrivee"] = np.where(
        df["circulations_effectives"] > 0,
        100 * retard / df["circulations_effectives"],
        np.nan,
    )
    df["taux_annulation"] = np.where(
        prevu > 0,
        100 * annule / prevu,
        np.nan,
    )
    df_valides = df[df["circulations_effectives"] > 0].copy()
    return df, df_valides


def taux_globaux(df_valides):
    """Calcule les taux globaux pondérés sur toutes les observations.

    Les totaux sont rapportés avant la moyenne afin que les liaisons à fort
    volume influencent davantage l'indicateur global.
    """
    retard = (
        100 * df_valides["nb_train_retard_arrivee"].sum()
        / df_valides["circulations_effectives"].sum()
    )
    annulation = (
        100 * df_valides["nb_annulation"].sum()
        / df_valides["nb_train_prevu"].sum()
    )
    return round(retard, 6), round(annulation, 6)


def comparaison_depart_arrivee(df_valides):
    """Compare chaque variable météo commune entre départ et arrivée.

    L'écart vaut toujours arrivée moins départ : une valeur positive indique
    que la mesure est plus élevée à l'arrivée.
    """
    bases = [
        "RR", "TN", "TX", "TAMPLIM",
        "NBJGELEE", "NBJRR10", "NBJORAG",
        "NBJBROU", "NBJFF16", "NBJTX30",
    ]
    resultats = []

    for variable in bases:
        depart = f"{variable}_depart"
        arrivee = f"{variable}_arrivee"
        if depart not in df_valides or arrivee not in df_valides:
            continue

        ecart = (
            pd.to_numeric(df_valides[arrivee], errors="coerce")
            - pd.to_numeric(df_valides[depart], errors="coerce")
        ).dropna()
        if ecart.empty:
            continue

        resultats.append({
            "variable": variable,
            "nombre_observations": len(ecart),
            "depart_moyen": df_valides[depart].mean(),
            "arrivee_moyenne": df_valides[arrivee].mean(),
            "ecart_moyen_arrivee_depart": ecart.mean(),
            "ecart_median_arrivee_depart": ecart.median(),
        })

    return pd.DataFrame(resultats).round(6)


def comparaison_par_tiers(df_valides):
    """Calcule les taux pondérés dans les tiers bas, moyen et haut.

    Chaque variable est découpée indépendamment en trois quantiles. Cette
    construction décrit des niveaux faibles, intermédiaires et élevés sans
    imposer des seuils météorologiques externes.
    """
    retards = []
    annulations = []

    for variable in VARIABLES_METEO:
        if variable not in df_valides:
            continue

        sous_base = df_valides[
            [
                variable,
                "nb_train_prevu",
                "nb_annulation",
                "circulations_effectives",
                "nb_train_retard_arrivee",
            ]
        ].copy()
        sous_base[variable] = pd.to_numeric(
            sous_base[variable], errors="coerce"
        )
        sous_base = sous_base.dropna()
        if sous_base.empty:
            continue

        codes = pd.qcut(
            sous_base[variable],
            q=3,
            labels=False,
            duplicates="drop",
        )
        nombre_tiers = pd.Series(codes, index=sous_base.index).nunique()
        sous_base["tiers"] = codes.map(
            lambda code:
                "Tiers bas" if code == 0
                else "Tiers haut" if code == nombre_tiers - 1
                else "Tiers moyen"
        )

        groupes = sous_base.groupby("tiers", observed=True).agg(
            observations=("tiers", "size"),
            trains_prevus=("nb_train_prevu", "sum"),
            annulations=("nb_annulation", "sum"),
            circulations=("circulations_effectives", "sum"),
            retards=("nb_train_retard_arrivee", "sum"),
        )
        groupes["taux_retard"] = (
            100 * groupes["retards"] / groupes["circulations"]
        )
        groupes["taux_annulation"] = (
            100 * groupes["annulations"] / groupes["trains_prevus"]
        )

        for tiers, ligne in groupes.iterrows():
            retards.append({
                "variable": variable,
                "tiers": tiers,
                "observations": ligne["observations"],
                "taux_pondere_pct": ligne["taux_retard"],
            })
            annulations.append({
                "variable": variable,
                "tiers": tiers,
                "observations": ligne["observations"],
                "taux_pondere_pct": ligne["taux_annulation"],
            })

    return (
        pd.DataFrame(retards).round(6),
        pd.DataFrame(annulations).round(6),
    )


def couverture_jointure(df, df_valides):
    """Mesure la taille et la couverture complète des jointures."""
    lignes = [
        {
            "indicateur": "Observations jointes",
            "valeur": len(df),
        },
        {
            "indicateur": "Observations avec circulations effectives positives",
            "valeur": len(df_valides),
        },
        {
            "indicateur": "Part conservée pour les taux (%)",
            "valeur": round(len(df_valides) / len(df) * 100, 2),
        },
    ]
    for cote in ["depart", "arrivee"]:
        variables_principales = [
            c for c in [
                f"RR_{cote}",
                f"TN_{cote}",
                f"TX_{cote}",
                f"NBJTX30_{cote}",
            ]
            if c in df.columns
        ]
        if variables_principales:
            couverture = df[variables_principales].notna().all(
                axis=1
            ).mean() * 100
            lignes.append({
                "indicateur": f"Couverture météo principale {cote} (%)",
                "valeur": round(couverture, 2),
            })
    return pd.DataFrame(lignes)


def profils_liaisons(df_valides, nombre=15):
    """Classe les liaisons selon leurs taux pondérés.

    Les observations, circulations et trains prévus accompagnent les taux
    afin d'éviter d'interpréter une liaison au faible volume comme aussi
    représentative qu'une liaison très fréquentée.
    """
    liaisons = df_valides.groupby(
        ["gare_depart", "gare_arrivee"],
        dropna=False,
    ).agg(
        observations=("date", "size"),
        circulations=("circulations_effectives", "sum"),
        retards=("nb_train_retard_arrivee", "sum"),
        trains_prevus=("nb_train_prevu", "sum"),
        annulations=("nb_annulation", "sum"),
    ).reset_index()
    liaisons["taux_retard"] = (
        100 * liaisons["retards"] / liaisons["circulations"]
    )
    liaisons["taux_annulation"] = (
        100 * liaisons["annulations"] / liaisons["trains_prevus"]
    )
    liaisons = liaisons[
        liaisons["circulations"] > 0
    ].sort_values("taux_retard", ascending=False)
    return liaisons.head(nombre).round(6)


def generer_stats_jointe(input_path="df_sncf_avec_meteo.csv"):
    """API unique appelée par le notebook."""
    df, df_valides = charger_base(input_path)
    retards, annulations = comparaison_par_tiers(df_valides)

    return {
        "data": df,
        "data_valides": df_valides,
        "ecarts_depart_arrivee": comparaison_depart_arrivee(df_valides),
        "couverture": couverture_jointure(df, df_valides),
        "profils_liaisons": profils_liaisons(df_valides),
        "retards_par_tiers": retards,
        "annulations_par_tiers": annulations,
    }
