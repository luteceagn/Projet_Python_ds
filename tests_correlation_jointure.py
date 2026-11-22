"""Correlations, covariance et redondances de la base SNCF-Meteo jointe."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


VARIABLES_METEO = [
    "RR_depart", "TN_depart", "TX_depart", "TAMPLIM_depart",
    "NBJGELEE_depart", "NBJRR10_depart",
    "NBJORAG_depart", "NBJBROU_depart", "NBJFF16_depart", "NBJTX30_depart",
    "RR_arrivee", "TN_arrivee", "TX_arrivee", "TAMPLIM_arrivee",
    "NBJGELEE_arrivee", "NBJRR10_arrivee",
    "NBJORAG_arrivee", "NBJBROU_arrivee", "NBJFF16_arrivee", "NBJTX30_arrivee",
]

VARIABLES_SNCF = [
    "circulations_effectives",
    "taux_retard_arrivee",
    "taux_annulation",
    "nb_annulation",
    "nb_train_retard_arrivee",
]


def charger_base(input_path="df_sncf_avec_meteo.csv"):
    """Charge la jointure et reconstruit les indicateurs SNCF utilisés."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Base jointe introuvable : {path}")

    df = pd.read_csv(path)
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


def tester_correlations(df_valides):
    """Test Pearson linéaire et Spearman monotone pour chaque couple.

    Les p-valeurs mesurent la significativité du coefficient observé.
    Elles ne démontrent pas une causalité ni une taille d'effet importante.
    """
    lignes = []

    for variable_sncf in VARIABLES_SNCF:
        for variable_meteo in VARIABLES_METEO:
            if variable_meteo not in df_valides:
                continue

            ensemble = pd.concat([
                pd.to_numeric(df_valides[variable_meteo], errors="coerce"),
                pd.to_numeric(df_valides[variable_sncf], errors="coerce"),
            ], axis=1).dropna()
            if len(ensemble) < 3:
                continue

            x = ensemble.iloc[:, 0]
            y = ensemble.iloc[:, 1]
            if x.nunique() < 2 or y.nunique() < 2:
                continue

            pearson, p_pearson = pearsonr(x, y)
            spearman, p_spearman = spearmanr(x, y)
            lignes.append({
                "variable_sncf": variable_sncf,
                "variable_meteo": variable_meteo,
                "n": len(ensemble),
                "pearson": pearson,
                "p_pearson": p_pearson,
                "spearman": spearman,
                "p_spearman": p_spearman,
            })

    return pd.DataFrame(lignes).round({
        "pearson": 6,
        "p_pearson": 8,
        "spearman": 6,
        "p_spearman": 8,
    })


def calculer_matrices(df_valides):
    """Construit la covariance puis la corrélation normalisée.

    La covariance conserve les unités propres des variables. La corrélation
    est comprise entre -1 et 1 et facilite les comparaisons entre variables.
    """
    variables = VARIABLES_METEO + [
        "taux_retard_arrivee",
        "taux_annulation",
    ]
    variables = [v for v in variables if v in df_valides]
    numeriques = df_valides[variables].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return numeriques.cov(min_periods=30), numeriques.corr(method="pearson")


def redondances_meteo(df_valides):
    """Repère les couples météo dont |r| dépasse 0,75.

    Ce seuil signale une redondance potentielle pour une ACP ou un
    clustering, sans imposer automatiquement la suppression d'une variable.
    """
    variables = [v for v in VARIABLES_METEO if v in df_valides]
    matrice = df_valides[variables].corr(method="pearson")
    resultats = []

    for i, variable in enumerate(matrice.columns):
        for autre in matrice.columns[i + 1:]:
            coefficient = matrice.loc[variable, autre]
            if abs(coefficient) >= 0.75:
                resultats.append({
                    "variable_1": variable,
                    "variable_2": autre,
                    "correlation": coefficient,
                })

    return pd.DataFrame(resultats).round(6)


def tracer_heatmap_cibles(matrice, output_path=None):
    """Visualise les cibles SNCF en corrélation avec les variables météo."""
    cible = matrice.loc[
        ["taux_retard_arrivee", "taux_annulation"]
    ]
    fig, ax = plt.subplots(figsize=(16, 4))
    image = ax.imshow(cible, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cible.columns)))
    ax.set_xticklabels(cible.columns, rotation=70, ha="right")
    ax.set_yticks(range(len(cible.index)))
    ax.set_yticklabels(cible.index)

    for i in range(cible.shape[0]):
        for j in range(cible.shape[1]):
            ax.text(j, i, f"{cible.iloc[i, j]:.2f}",
                    ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Corrélation Pearson")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def generer_tests_correlation(input_path="df_sncf_avec_meteo.csv",
                              output_dir="sorties_correlation_jointure"):
    """API unique appelée par le notebook."""
    df, df_valides = charger_base(input_path)
    tests = tester_correlations(df_valides)
    covariance, correlation = calculer_matrices(df_valides)
    redondances = redondances_meteo(df_valides)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tracer_heatmap_cibles(
            correlation,
            output_dir / "correlations_sncf_meteo_heatmap.png",
        )

    return {
        "data": df,
        "data_valides": df_valides,
        "tests": tests,
        "matrice_covariance": covariance,
        "matrice_correlation": correlation,
        "redondances": redondances,
    }
