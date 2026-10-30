import pandas as pd
import requests
from io import BytesIO


def charger_gares():

    # URL de l'API SNCF : gares de voyageurs
    url = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/gares-de-voyageurs/exports/csv"

    # Récupération des données
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    # Lecture directement en mémoire
    df_gares = pd.read_csv(
        BytesIO(response.content),
        sep=";",
        dtype={"codeinsee": "string"}
    )

    return df_gares


# Vérification
if __name__ == "__main__":

    df_gares = charger_gares()

    print("Dimensions :", df_gares.shape)
    print("Colonnes :", df_gares.columns.tolist())
    print(df_gares.head())