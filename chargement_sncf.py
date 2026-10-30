import pandas as pd
import requests
from io import BytesIO


def charger_sncf(date_debut="2018-01-01", date_fin="2026-09-30"):

    # URL de l'API SNCF
    url = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-tgv-aqst/exports/csv"

    # Paramètres de la requête
    params = {
        "where": f"date >= '{date_debut}' AND date <= '{date_fin}'",
        "sort": "date"
    }

    # Récupération des données
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    # Lecture du CSV directement en mémoire
    df_sncf = pd.read_csv(
        BytesIO(response.content),
        sep=";"
    )

    # Conversion de la date
    df_sncf["date"] = pd.to_datetime(df_sncf["date"])

    return df_sncf