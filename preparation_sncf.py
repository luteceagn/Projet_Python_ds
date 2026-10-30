import pandas as pd
import requests
from io import BytesIO

# 1. Période d'étude
date_debut = "2018-01-01"
date_fin = "2026-09-30"

# 2. URL de l'API SNCF
url = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-tgv-aqst/exports/csv"

# 3. Paramètres de la requête
params = {
    "where": f"date >= '{date_debut}' AND date <= '{date_fin}'",
    "sort": "date"
}

# 4. Récupération des données
response = requests.get(url, params=params, timeout=120)
response.raise_for_status()

# 5. Lecture des données directement en mémoire
df_sncf = pd.read_csv(
    BytesIO(response.content),
    sep=";"
)

# 6. Conversion de la date
df_sncf["date"] = pd.to_datetime(df_sncf["date"])

# 7. Vérification des données
print("Dimensions :", df_sncf.shape)
print("Première date :", df_sncf["date"].min())
print("Dernière date :", df_sncf["date"].max())

print(df_sncf.head())