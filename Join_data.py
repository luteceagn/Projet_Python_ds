import pandas as pd
import numpy as np

# 1. Chargement des deux bases
df_sncf = pd.read_csv("df_sncf_prepare.csv")
df_meteo = pd.read_csv("data_meteo_enrichie.csv", sep=";")

# 2. Trie de variables et créations des nouvelles
#   a. On ne garde que les observations qualitatives (inférieure à 2)
# Prise en compte de la qualité (si la qualité est >= 2, on met la valeur à NaN)
# Mettre la valeur à NaN ne gènera pas Pandas plus tard et permet de garder des données de qualité pour l'aggrégation
# Pour les précipitations
if 'QRR' in df_meteo.columns:
    df_meteo.loc[pd.to_numeric(df_meteo['QRR'], errors='coerce') == 2, 'RR'] = np.nan

# Pour la température minimale
if 'QTN' in df_meteo.columns:
    df_meteo.loc[pd.to_numeric(df_meteo['QTN'], errors='coerce') == 2, 'TN'] = np.nan

# Pour la température maximale
if 'QTX' in df_meteo.columns:
    df_meteo.loc[pd.to_numeric(df_meteo['QTX'], errors='coerce') == 2, 'TX'] = np.nan

# Pour l'amplitude thermique
if 'QTAMPLIM' in df_meteo.columns:
    df_meteo.loc[pd.to_numeric(df_meteo['QTAMPLIM'], errors='coerce') == 2, 'TAMPLIM'] = np.nan

#   b. Calcul des anomalies
# Pour la pluie : pourcentage d'écart par rapport à la normale (en gérant les cas où RR_ME vaut 0)
df_meteo['anomalie_rr_pct'] = np.where(
    df_meteo['RR_ME'] > 0,
    ((df_meteo['RR'] - df_meteo['RR_ME']) / df_meteo['RR_ME']) * 100,
    0
)

# Pour les températures : écart absolu en degrés Celsius
df_meteo['anomalie_tn'] = df_meteo['TN'] - df_meteo['TN_ME']
df_meteo['anomalie_tx'] = df_meteo['TX'] - df_meteo['TX_ME']

#   c. Sélection des variables à conserver (clés de jointure + variables cibles)
variables_utiles = [
    'departement',
    'DATE',

    # Froid, neige et minimales
    'NBJGELEE', # Nombre de jours de gel (TN≤0"°C" )
    'NBJSOLNG', # Nombre de jours avec sol enneigé
    'TN', # Moyenne mensuelle des températures minimales quotidiennes
    'anomalie_tn', # Différence de degré comparé à la normale climatologique

    # Pluie, orages et brouillard
    'RR', # Cumul mensuel des précipitations (en mm)
    'NBJRR10', # Nombre de jours de fortes pluies (>= 10mm)
    'anomalie_rr_pct', # Différence en % de précipitations comparé à la normale climatologique
    'NBJORAG', # Nombre de jours d'orage dans le mois
    'NBJBROU', # Nombre de jours de brouillard dans le mois
    # Vent
    'NBJFF16', # Nombre de jours avec vent moyen dépassant 16 m/s

    # Chaleur et maximales
    'TX', # Moyenne mensuelle des températures maximales quotidiennes
    'anomalie_tx', # Différence en degré comparé à la normale climatologique
    'NBJTX30', # Nombre de jours de chaleur (TX≥30"°C" )

    # Amplitude thermique
    'TAMPLIM' # Amplitude thermique moyenne du mois
]

# Filtrage du DataFrame
df_meteo = df_meteo[variables_utiles].copy()

# 2. Calcule de la moyenne de toutes les colonnes numériques par département et par mois
# numeric_only=True évite d'essayer de faire la moyenne sur le nom des stations (texte)
df_meteo_agg = df_meteo.groupby(['departement', 'DATE']).mean(numeric_only=True).reset_index()

# 1. Transformer la colonne DATE en objet Period pour récupérer les jours du mois
df_meteo_agg['nb_jours_mois'] = pd.to_datetime(df_meteo_agg['DATE'].astype(str), format='%Y%m').dt.days_in_month

# 2. Transformer les compteurs de phénomène en pourcentages de jours par mois
colonnes_comptage = ['NBJGELEE', 'NBJSOLNG', 'NBJRR10', 'NBJORAG', 'NBJBROU','NBJFF16', 'NBJTX30']

for col in colonnes_comptage:
    df_meteo_agg[col] = (df_meteo_agg[col] / df_meteo_agg['nb_jours_mois']) * 100
df_meteo_agg = df_meteo_agg.drop(columns=['nb_jours_mois'])

# On arrondit les variables numériques à 0 décimales après la virgule pour les données météo et 1 décimale pour les données sncf
df_meteo_agg = df_meteo_agg.round(0)
colonnes_numeriques = df_sncf.select_dtypes(include=['float64', 'float32']).columns
df_sncf[colonnes_numeriques] = df_sncf[colonnes_numeriques].round(1)


# 3. Préparer la clé de mois dans la base SNCF
# df_sncf['date'] est du type 'YYYY-MM-DD' -> on extrait l'année et le mois pour obtenir '201801' (en entier ou string)
df_sncf['annee_mois'] = pd.to_datetime(df_sncf['date']).dt.strftime('%Y%m').astype(int)

# 4. S'assurer que le département météo et le département SNCF ont le même type (ex: string ou int)
df_meteo_agg['departement'] = df_meteo_agg['departement'].astype(str).str.zfill(2)
df_sncf['dep_depart'] = df_sncf['dep_depart'].astype(str).str.zfill(2)
df_sncf['dep_arrivee'] = df_sncf['dep_arrivee'].astype(str).str.zfill(2)
df_meteo_agg['DATE'] = df_meteo_agg['DATE'].astype(int)


# 5. Première jointure : Météo du département de DÉPART
# On renomme toutes les colonnes météo entrantes avec le suffixe '_depart' pour les différencier
meteo_depart = df_meteo_agg.add_suffix('_depart')
df_merged = pd.merge(
    df_sncf,
    meteo_depart,
    left_on=['dep_depart', 'annee_mois'],
    right_on=['departement_depart', 'DATE_depart'],
    how='left',
    validate='many_to_one'
)

# 6. Seconde jointure : Météo du département d'ARRIVÉE
# On renomme toutes les colonnes météo entrantes avec le suffixe '_arrivee'
meteo_arrivee = df_meteo_agg.add_suffix('_arrivee')
df_final = pd.merge(
    df_merged,
    meteo_arrivee,
    left_on=['dep_arrivee', 'annee_mois'],
    right_on=['departement_arrivee', 'DATE_arrivee'],
    how='left',
    validate='many_to_one'
)

# 7. Nettoyage des colonnes de jointure en double si besoin
df_final = df_final.drop(columns=['departement_depart', 'DATE_depart', 'departement_arrivee', 'DATE_arrivee','annee_mois'])

# Aperçu du résultat
print(df_final[['date', 'gare_depart', 'dep_depart', 'RR_depart', 'gare_arrivee', 'dep_arrivee', 'RR_arrivee']].head())

# Sauvegarde du résultat final
df_final.to_csv("df_sncf_avec_meteo.csv", index=False, encoding="utf-8")