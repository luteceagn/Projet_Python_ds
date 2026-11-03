import io
import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

# Charge les variables définies dans le fichier .env
load_dotenv()

API_KEY = os.getenv("METEO_FRANCE_API_KEY")
if not API_KEY:
  raise ValueError("Fichier .env non détecté ou clé introuvable !")

HEADERS = {"apikey": API_KEY, "accept": "application/json"}

# 💡 Testez d'abord avec un seul département (ex: 75) pour valider
DEPARTEMENTS = [75]

OUTPUT_FILE = "meteo_mensuelle_2018_2026.csv"

# Liste des années à parcourir (de 2018 à 2026 inclus)
ANNEES = range(2018, 2026)

all_dfs = []

for dept in DEPARTEMENTS:
  print(f"\n================ 📂 Département {dept} ================")

  url_stations = (
      "https://public-api.meteofrance.fr/public/DPClim/v1/liste-stations/quotidienne"
  )
  resp_stations = requests.get(
      url_stations, headers=HEADERS, params={"id-departement": int(dept)}
  )

  if resp_stations.status_code != 200:
    print(f"⚠️ Erreur récupération stations : {resp_stations.status_code}")
    continue

  stations = resp_stations.json()
  print(f"👉 {len(stations)} stations trouvées.")

  for station in stations:
    station_id = station.get("id") or station.get("ografica")
    if not station_id:
      continue

    station_id = str(station_id).zfill(8)

    # 🔄 Boucle sur chaque année (maximum 1 an par requête demandé par l'API)
    for annee in ANNEES:
      date_deb = f"{annee}-01-01T00:00:00Z"
      date_fin = f"{annee}-12-31T00:00:00Z"

      print(f"   [Station {station_id}] - Année {annee}...")

      url_commande = "https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/mensuelle"
      resp_cmd = requests.get(
          url_commande,
          headers=HEADERS,
          params={
              "id-station": station_id,
              "date-deb-periode": date_deb,
              "date-fin-periode": date_fin,
          },
      )

      if resp_cmd.status_code == 429:
        print("     ⏳ 429 Trop de requêtes ! Pause de 15 secondes...")
        time.sleep(15)
        continue
      elif resp_cmd.status_code != 202:
        # Année sans données pour cette station, on passe à l'année suivante
        print(f"     ❌ Erreur {resp_cmd.status_code} - Détail du serveur : {resp_cmd.text}")
        continue

      cmd_data = resp_cmd.json()
      id_cmde = (
          cmd_data.get("id-cmde")
          or cmd_data.get("elab")
          or list(cmd_data.values())[0]
      )

      url_fichier = (
          "https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier"
      )
      csv_content = None

      # Polling pour récupérer le fichier de l'année
      for _ in range(10):
        time.sleep(3)
        resp_file = requests.get(
            url_fichier, headers=HEADERS, params={"id-cmde": id_cmde}
        )

        if resp_file.status_code == 200:
          csv_content = resp_file.content.decode("utf-8", errors="replace")
          break
        elif resp_file.status_code == 429:
          time.sleep(10)
        elif resp_file.status_code in [201, 204]:
          continue
        else:
          break

      if csv_content:
        try:
          df_station = pd.read_csv(
              io.StringIO(csv_content), sep=";", low_memory=False
          )
          all_dfs.append(df_station)
        except Exception:
          pass

      # Petite pause de courtoisie pour éviter de saturer l'API
      time.sleep(1.5)

# --- FUSION FINALE ---
if all_dfs:
  print("\n🔄 Fusion de toutes les données dans le fichier unique...")
  master_df = pd.concat(all_dfs, ignore_index=True)
  master_df.to_csv(OUTPUT_FILE, index=False, sep=";")
  print(
      f"\n🎉 Succès total ! Fichier unique généré : {OUTPUT_FILE} avec"
      f" {len(master_df)} lignes."
  )
else:
  print("\n❌ Aucune donnée n'a pu être récupérée.")
