import glob
import pandas as pd

# ============================================================================
# ÉTAPE 1 : Concaténation des bases de données des 100+ départements de France
# ============================================================================

# Les fichiers doivent être des CSV situés dans un dossier nommé 'data_1950_2024/'
# Adaptez le chemin et l'extension (.csv, .xlsx, .parquet, etc.) si besoin
chemin_fichiers = 'data_1950_2024/*.csv'

# Récupère la liste de tous les fichiers correspondants
fichiers = glob.glob(chemin_fichiers)
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
        engine='python',
    )
    liste_dfs.append(df_temp)

  except Exception as e:
    print(f'⚠️ Erreur sur le fichier {f} : {e}')

if liste_dfs:
  df_global = pd.concat(liste_dfs, ignore_index=True)
  print(f'Fusion réussie ! Total de lignes : {len(df_global)}')
else:
  print("Aucun fichier n'a pu être chargé.")

# ==========================================
# ÉTAPE 2 : Filtrage par année (format yyyymm)
# ==========================================

colonne_date = 'AAAAMM'

# Extraction de chaîne de caractères
# Puisque l'on a beaucoup de lignes et que le format 'yyyymm' est strict,
# on peut simplement extraire les 4 premiers caractères (l'année) :
# On ne garde que les données des années 2018 à 2024 qui correspondent à notre base SNCF
annee_str = ['2018','2019','2020','2021','2022','2023','2024']
df_global['annee'] = df_global[colonne_date].astype(str).str[:4]
df_filtre = df_global[df_global['annee'].isin(annee_str)]


# ==========================================
# ÉTAPE 3 : Sauvegarde de la base finale
# ==========================================
df_filtre.to_csv('data_1950_2024.csv', index=False)
print(f'Nombre de lignes après filtrage : {len(df_filtre)}')
