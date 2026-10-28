import glob
import pandas as pd

# ============================================================================
# ÉTAPE 1 : Concaténation des bases de données des 100+ départements de France
# ============================================================================

# Les fichiers doivent être des CSV situés dans un dossier nommé 'data_2025_2026'
# Adaptez le chemin et l'extension (.csv, .xlsx, .parquet, etc.) si besoin
chemin_fichiers = 'data_2025_2026/*.csv'

# Récupération de la liste de tous les fichiers correspondants
fichiers = glob.glob(chemin_fichiers)
data = glob.glob('data_1950_2024.csv') # données concaténée précédemment
# pour réaliser la concaténation finale
liste_dfs = []

for f in fichiers:
  print(f'Lecture en cours : {f}')
  try:
    df_temp = pd.read_csv(
        f,
        sep=';',
        encoding='utf-8',
        quotechar='"',  # Gère les textes entre guillemets contenant des points-virgules
        on_bad_lines='skip',  # Ignore définitivement les lignes corrompues
        engine='python',  # Moteur Python (plus lent mais beaucoup plus tolérant que le moteur C)
    )
    liste_dfs.append(df_temp)

  except Exception as e:
    print(f'⚠️ Erreur sur le fichier {f} : {e}')

if liste_dfs:
  df_global = pd.concat(liste_dfs, ignore_index=True)
  print(f'Fusion réussie ! Total de lignes : {len(df_global)}')
else:
  print("Aucun fichier n'a pu être chargé.")

# ===============================================================
# Etape 2 : Création de la bdd complète
# ===============================================================
# 1. Chargement du fichier CSV supplémentaire sous forme de DataFrame
df_supplementaire = pd.read_csv(
    'data_1950_2024.csv',
    sep=';',
    on_bad_lines='skip',
    encoding='utf-8',
)

# 2. Ajout à la fin du df_global existant

df_global = pd.concat([df_global, df_supplementaire], ignore_index=True)

print(f'Base finale mise à jour ! Total de lignes : {len(df_global)}')
# ===============================================================
# ÉTAPE 3 : Sauvegarde de la base finale des données de 2025-2026
# ===============================================================
df_global.to_csv('data_meteo.csv', index=False)
print(f'Nombre de lignes après filtrage : {len(df_global)}')
