"""Corrélations, covariance et redondances de la base SNCF-Météo jointe (partie 6).

Le module est appelé par le rapport via ``generer_tests_correlation`` et
fournit :
- les tests de corrélation de Spearman SNCF x météo (6.2) ;
- la matrice de Spearman 3xn SNCF x météo et sa heatmap (6.3) ;
- la matrice de variance-covariance nxn préparant une ACP (6.4) ;
- les couples de variables météo redondantes (6.5) ;
- la matrice de Spearman nxn complète (non affichée, conservée pour l'ACP).

Le chargement de la base et la liste des variables météo sont repris de
``stats_desc_jointe`` afin que les parties 5.2 et 6 utilisent exactement les
mêmes indicateurs (en pourcentage).
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from stats_desc_jointe import VARIABLES_METEO, charger_base


# Indicateurs SNCF en pourcentage, construits par ``charger_base``.
VARIABLES_SNCF = [
    "pourcentage_retard_depart",
    "pourcentage_retard_arrivee",
    "pourcentage_annulation",
]

# Seuil de |rho| à partir duquel deux variables météo sont jugées redondantes.
SEUIL_REDONDANCE = 0.75


def _numeriques(df_valides, variables):
    """Extrait les colonnes présentes dans la base, converties en nombres."""
    variables = [v for v in variables if v in df_valides]
    return df_valides[variables].apply(pd.to_numeric, errors="coerce")


def tester_correlations(df_valides):
    """Teste la corrélation de Spearman pour chaque couple SNCF x météo (6.2).

    Spearman travaille sur les rangs : il mesure une relation monotone et
    reste peu sensible aux valeurs extrêmes et aux distributions asymétriques.
    Les valeurs manquantes sont retirées couple par couple. Les p-valeurs ne
    démontrent ni causalité ni importance pratique.
    """
    lignes = []

    for variable_sncf in VARIABLES_SNCF:
        for variable_meteo in VARIABLES_METEO:
            if variable_meteo not in df_valides:
                continue

            couple = _numeriques(df_valides, [variable_meteo, variable_sncf]).dropna()
            x = couple[variable_meteo]
            y = couple[variable_sncf]
            # Le coefficient n'est pas défini si l'une des séries est constante.
            if len(couple) < 3 or x.nunique() < 2 or y.nunique() < 2:
                continue

            spearman, p_spearman = spearmanr(x, y)
            lignes.append({
                "variable_sncf": variable_sncf,
                "variable_meteo": variable_meteo,
                "n": len(couple),
                "spearman": spearman,
                "p_spearman": p_spearman,
            })

    # Les p-valeurs ne sont pas arrondies : avec n > 11 000, elles sont
    # souvent inférieures à 1e-8 et un arrondi les ramènerait toutes à 0.
    return pd.DataFrame(lignes).round({"spearman": 6})


def correlation_sncf_meteo(df_valides):
    """Construit la matrice de Spearman 3xn : indicateurs SNCF x météo (6.3)."""
    numeriques = _numeriques(df_valides, VARIABLES_SNCF + VARIABLES_METEO)
    variables_sncf = [v for v in VARIABLES_SNCF if v in numeriques]
    variables_meteo = [v for v in VARIABLES_METEO if v in numeriques]
    return numeriques.corr(method="spearman").loc[variables_sncf, variables_meteo]


def calculer_correlation_spearman(df_valides):
    """Construit la matrice de corrélation de Spearman nxn.

    Elle porte sur les mêmes 23 variables que la matrice de covariance (20
    variables météo puis 3 indicateurs SNCF). Bornée entre -1 et 1, elle
    permet de comparer les associations sans dépendre des unités. Elle n'est
    pas affichée dans le rapport, mais est conservée pour une future ACP ou un
    clustering.
    """
    numeriques = _numeriques(df_valides, VARIABLES_METEO + VARIABLES_SNCF)
    return numeriques.corr(method="spearman")


def calculer_covariance(df_valides):
    """Construit la matrice de variance-covariance nxn (6.4).

    Elle porte sur les 20 variables météo puis les 3 indicateurs SNCF, soit le
    tableau d'une future ACP. La covariance conserve les unités propres des
    variables : la diagonale contient leurs variances. Chaque case est
    calculée sur les observations disponibles pour le couple, avec au moins
    30 observations.
    """
    numeriques = _numeriques(df_valides, VARIABLES_METEO + VARIABLES_SNCF)
    return numeriques.cov(min_periods=30)


def redondances_meteo(df_valides, seuil=SEUIL_REDONDANCE):
    """Repère les couples météo dont la corrélation de Spearman |rho| >= seuil (6.5).

    Ce seuil signale une redondance potentielle pour une ACP ou un
    clustering, sans imposer automatiquement la suppression d'une variable.
    """
    matrice = _numeriques(df_valides, VARIABLES_METEO).corr(method="spearman")
    resultats = []

    # Parcours du seul triangle supérieur pour ne lister chaque couple qu'une fois.
    for i, variable in enumerate(matrice.columns):
        for autre in matrice.columns[i + 1:]:
            coefficient = matrice.loc[variable, autre]
            if abs(coefficient) >= seuil:
                resultats.append({
                    "variable_1": variable,
                    "variable_2": autre,
                    "correlation": coefficient,
                })

    return pd.DataFrame(resultats).round(6)


def sauvegarder_heatmap_spearman(correlation_sncf, dossier):
    """Enregistre la heatmap de la matrice de Spearman SNCF x météo (6.3)."""

    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)

    n_lignes, n_colonnes = correlation_sncf.shape
    largeur = max(13, n_colonnes * 0.65)
    hauteur = max(3.2, n_lignes * 0.95 + 1.8)

    fig, ax = plt.subplots(figsize=(largeur, hauteur))
    sns.heatmap(
        correlation_sncf,
        cmap="coolwarm",
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8.5, "fontweight": "bold"},
        cbar_kws={"shrink": 0.85, "pad": 0.02},
        ax=ax,
    )

    ax.set_title(
        "Corrélation de Spearman : SNCF et météo",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    # Légendes perpendiculaires : X en vertical, Y en horizontal
    ax.tick_params(axis="x", labelsize=9, rotation=90)
    ax.tick_params(axis="y", labelsize=9.5, rotation=0)

    fig.tight_layout()
    fig.savefig(
        dossier / "correlation_spearman_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def generer_tests_correlation(input_path="df_sncf_avec_meteo.csv", output_dir=None):
    """Point d'entrée appelé par le rapport pour la partie 6.

    Si ``output_dir`` est renseigné, la heatmap de Spearman affichée en 6.3 y
    est enregistrée. Retourne un dictionnaire contenant la base des
    observations valides et les résultats des sous-parties 6.2 à 6.5.
    """
    _, df_valides = charger_base(input_path)
    correlation_sncf = correlation_sncf_meteo(df_valides)

    if output_dir is not None:
        sauvegarder_heatmap_spearman(correlation_sncf, output_dir)

    return {
        "data_valides": df_valides,
        "tests": tester_correlations(df_valides),
        "correlation_sncf_meteo": correlation_sncf,
        "matrice_correlation": calculer_correlation_spearman(df_valides),
        "matrice_covariance": calculer_covariance(df_valides),
        "redondances": redondances_meteo(df_valides),
    }
