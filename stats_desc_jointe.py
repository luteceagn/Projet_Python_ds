"""Statistiques descriptives de la base SNCF-Météo jointe (partie 5.2).

Le module est appelé par le rapport via ``generer_stats_jointe`` et fournit :
- la couverture de la jointure (5.2.1.1) ;
- les liaisons au plus fort pourcentage de retard (5.2.2) ;
- la comparaison des variables météo entre départ et arrivée (5.2.3) ;
- les pourcentages de retard et d'annulation de part et d'autre de la
  médiane de chaque variable météo (5.2.4).

Il fournit aussi ``charger_base`` et ``VARIABLES_METEO``, réutilisés par
``tests_correlation_jointure`` (partie 6).

Tous les indicateurs SNCF sont exprimés en pourcentage (taux x 100).
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Variables météo mensuelles retenues, mesurées dans le département de la
# gare de départ puis dans celui de la gare d'arrivée.
VARIABLES_METEO = [
    "RR_depart", "TN_depart", "TX_depart", "TAMPLIM_depart",
    "NBJGELEE_depart", "NBJRR10_depart",
    "NBJORAG_depart", "NBJBROU_depart", "NBJFF16_depart", "NBJTX30_depart",
    "RR_arrivee", "TN_arrivee", "TX_arrivee", "TAMPLIM_arrivee",
    "NBJGELEE_arrivee", "NBJRR10_arrivee",
    "NBJORAG_arrivee", "NBJBROU_arrivee", "NBJFF16_arrivee", "NBJTX30_arrivee",
]


def charger_base(input_path="df_sncf_avec_meteo.csv"):
    """Charge la jointure et construit les indicateurs SNCF en pourcentage.

    Les circulations effectives valent les trains prévus moins les
    annulations. Les pourcentages de retard (au départ et à l'arrivée) sont
    rapportés aux circulations effectives, celui d'annulation aux trains
    prévus.

    Retourne la base complète et la sous-base des observations ayant des
    circulations effectives positives (seules utilisables pour les
    pourcentages de retard).
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Base jointe introuvable : {path}. "
            "Exécutez d'abord la jointure de la partie 5 du rapport."
        )

    df = pd.read_csv(path)

    # Variables calendaires (non utilisées pour l'instant, conservées pour
    # d'éventuelles analyses par année ou par saison).
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["annee"] = df["date"].dt.year
    df["mois"] = df["date"].dt.month

    # Conversion défensive : une valeur non numérique devient NaN.
    prevu = pd.to_numeric(df["nb_train_prevu"], errors="coerce")
    annule = pd.to_numeric(df["nb_annulation"], errors="coerce")
    retard_depart = pd.to_numeric(df["nb_train_depart_retard"], errors="coerce")
    retard_arrivee = pd.to_numeric(df["nb_train_retard_arrivee"], errors="coerce")

    df["circulations_effectives"] = prevu - annule
    circulations_positives = df["circulations_effectives"] > 0

    # Les pourcentages sont laissés à NaN lorsque le dénominateur est nul.
    df["pourcentage_retard_arrivee"] = np.where(
        circulations_positives,
        100 * retard_arrivee / df["circulations_effectives"],
        np.nan,
    )
    df["pourcentage_retard_depart"] = np.where(
        circulations_positives,
        100 * retard_depart / df["circulations_effectives"],
        np.nan,
    )
    df["pourcentage_annulation"] = np.where(
        prevu > 0,
        100 * annule / prevu,
        np.nan,
    )

    df_valides = df[circulations_positives].copy()
    return df, df_valides


def couverture_jointure(df, df_valides):
    """Mesure la taille de la jointure et la couverture météo (5.2.1.1).

    La couverture d'une extrémité est la part des lignes où les quatre
    variables météo principales (RR, TN, TX, NBJTX30) sont toutes renseignées.
    """
    lignes = [
        {"indicateur": "Observations jointes", "valeur": len(df)},
        {
            "indicateur": "Observations avec circulations effectives positives",
            "valeur": len(df_valides),
        },
        {
            "indicateur": "Part conservée pour les pourcentages (%)",
            "valeur": round(len(df_valides) / len(df) * 100, 2),
        },
    ]

    for cote in ["depart", "arrivee"]:
        variables_principales = [
            c for c in [f"RR_{cote}", f"TN_{cote}", f"TX_{cote}", f"NBJTX30_{cote}"]
            if c in df.columns
        ]
        if variables_principales:
            couverture = df[variables_principales].notna().all(axis=1).mean() * 100
            lignes.append({
                "indicateur": f"Couverture météo principale {cote} (%)",
                "valeur": round(couverture, 2),
            })

    return pd.DataFrame(lignes)


def profils_liaisons(df_valides, nombre=15):
    """Classe les liaisons selon leur pourcentage pondéré de retard (5.2.2).

    Les pourcentages sont calculés à partir des totaux de chaque liaison sur
    toute la période (pondération par le volume de trains). Les volumes
    (observations, circulations, trains prévus) sont conservés pour ne pas
    interpréter une liaison peu fréquentée comme aussi représentative qu'une
    liaison très fréquentée.
    """
    liaisons = df_valides.groupby(
        ["gare_depart", "gare_arrivee"], dropna=False
    ).agg(
        observations=("date", "size"),
        circulations=("circulations_effectives", "sum"),
        retards=("nb_train_retard_arrivee", "sum"),
        retards_depart=("nb_train_depart_retard", "sum"),
        trains_prevus=("nb_train_prevu", "sum"),
        annulations=("nb_annulation", "sum"),
    ).reset_index()
    liaisons = liaisons[liaisons["circulations"] > 0].copy()

    liaisons["pourcentage_retard_depart"] = (
        100 * liaisons["retards_depart"] / liaisons["circulations"]
    )
    liaisons["pourcentage_retard_arrivee"] = (
        100 * liaisons["retards"] / liaisons["circulations"]
    )
    liaisons["pourcentage_annulation"] = (
        100 * liaisons["annulations"] / liaisons["trains_prevus"]
    )

    liaisons = liaisons.sort_values("pourcentage_retard_arrivee", ascending=False)
    return liaisons.head(nombre).round(6)


def comparaison_depart_arrivee(df_valides):
    """Compare chaque variable météo entre départ et arrivée (5.2.3).

    L'écart vaut toujours arrivée moins départ : une valeur positive indique
    que la mesure est plus élevée à l'arrivée. Il est calculé sur les seules
    lignes où les deux extrémités sont renseignées.
    """
    # Noms de base communs aux deux extrémités : RR, TN, TX, ...
    bases = [
        v.removesuffix("_depart") for v in VARIABLES_METEO if v.endswith("_depart")
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


# Libellés des deux groupes formés autour de la médiane (5.2.4).
GROUPE_BAS = "≤ médiane"
GROUPE_HAUT = "> médiane"


def comparaison_par_mediane(df_valides):
    """Calcule les pourcentages pondérés de part et d'autre de la médiane (5.2.4).

    Chaque variable est coupée indépendamment en deux groupes autour de sa
    médiane, sans imposer de seuil météorologique externe : valeurs
    inférieures ou égales à la médiane d'un côté, strictement supérieures de
    l'autre. Les valeurs égales à la médiane rejoignent le groupe bas, si bien
    que les deux groupes peuvent être de tailles inégales lorsque beaucoup de
    valeurs sont identiques (par exemple NBJTX30, dont la médiane vaut 0).

    Retourne deux tables longues (variable, groupe) : pourcentage de retard à
    l'arrivée et pourcentage d'annulation, avec la médiane utilisée.
    """
    retards = []
    annulations = []

    for variable in VARIABLES_METEO:
        if variable not in df_valides:
            continue

        sous_base = df_valides[[
            variable,
            "nb_train_prevu",
            "nb_annulation",
            "circulations_effectives",
            "nb_train_retard_arrivee",
        ]].copy()
        sous_base[variable] = pd.to_numeric(sous_base[variable], errors="coerce")
        sous_base = sous_base.dropna()
        if sous_base.empty:
            continue

        # Coupure à la médiane : le groupe haut est vide seulement si la
        # médiane égale le maximum (variable quasi constante), cas ignoré.
        mediane = sous_base[variable].median()
        if not (sous_base[variable] > mediane).any():
            continue
        sous_base["groupe"] = np.where(
            sous_base[variable] > mediane, GROUPE_HAUT, GROUPE_BAS
        )

        # Pourcentages pondérés : rapport des totaux de chaque groupe.
        groupes = sous_base.groupby("groupe").agg(
            observations=("groupe", "size"),
            trains_prevus=("nb_train_prevu", "sum"),
            annulations=("nb_annulation", "sum"),
            circulations=("circulations_effectives", "sum"),
            retards=("nb_train_retard_arrivee", "sum"),
        )
        groupes["pourcentage_retard_arrivee"] = (
            100 * groupes["retards"] / groupes["circulations"]
        )
        groupes["pourcentage_annulation"] = (
            100 * groupes["annulations"] / groupes["trains_prevus"]
        )

        for groupe, ligne in groupes.iterrows():
            commun = {
                "variable": variable,
                "mediane": mediane,
                "groupe": groupe,
                "observations": ligne["observations"],
            }
            retards.append({
                **commun,
                "pourcentage_retard_arrivee": ligne["pourcentage_retard_arrivee"],
            })
            annulations.append({
                **commun,
                "pourcentage_annulation": ligne["pourcentage_annulation"],
            })

    return pd.DataFrame(retards).round(6), pd.DataFrame(annulations).round(6)


def generer_stats_jointe(input_path="df_sncf_avec_meteo.csv"):
    """Point d'entrée appelé par le rapport pour la partie 5.2.

    Retourne un dictionnaire contenant la base complète (``data``), la base
    des observations valides (``data_valides``) et les tableaux de résultats
    utilisés dans les sous-parties 5.2.1 à 5.2.4.
    """
    df, df_valides = charger_base(input_path)
    retards, annulations = comparaison_par_mediane(df_valides)

    return {
        "data": df,
        "data_valides": df_valides,
        "couverture": couverture_jointure(df, df_valides),
        "profils_liaisons": profils_liaisons(df_valides),
        "ecarts_depart_arrivee": comparaison_depart_arrivee(df_valides),
        "retards_par_mediane": retards,
        "annulations_par_mediane": annulations,
    }
