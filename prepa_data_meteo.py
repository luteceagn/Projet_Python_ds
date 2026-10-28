import pandas as pd

df = pd.read_csv(
    "data_meteo.csv", sep=',', on_bad_lines='skip', encoding='utf-8'
)

# Nettoie les espaces ou guillemets invisibles éventuels autour des noms de colonnes
df.columns = df.columns.str.strip().str.replace('"', '')

# Vérification : affiche la vraie liste de toutes les colonnes séparées
print('Colonnes chargées avec succès :', df.columns.tolist())
# 2. Extraction intelligente du département (2 ou 3 caractères)
# On convertit la colonne en texte
num_str = df['NUM_POSTE'].astype(str)

# Par défaut, on prend les 2 premiers caractères
df['DEP'] = num_str.str[:2]

# Si le code commence par '97' ou '98' (DROM), on prend les 3 premiers caractères
masque_dom_tom = num_str.str.startswith(('97', '98'))
df.loc[masque_dom_tom, 'DEP'] = num_str.str[:3]


# 3. Dictionnaire de correspondance
correspondance = {
    '01': 'Ain',
    '02': 'Aisne',
    '03': 'Allier',
    '04': 'Alpes-de-Haute-Provence',
    '05': 'Hautes-Alpes',
    '06': 'Alpes_Maritimes',
    '07': 'Ardèche',
    '08': 'Ardennes',
    '09': 'Ariège',
    '10': 'Aube',
    '11': 'Aude',
    '12': 'Aveyron',
    '13': 'Bouches-du-Rhône',
    '14': 'Calvados',
    '15': 'Cantal',
    '16': 'Charente',
    '17': 'Charente-Maritime',
    '18': 'Cher',
    '19': 'Corrèze',
    '2A': 'Corse-du-Sud',
    '2B': 'Haute-Corse',
    '21': "Côte-d'Or",
    '22': "Côtes-d'Armor",
    '23': 'Creuse',
    '24': 'Dordogne',
    '25': 'Doubs',
    '26': 'Drôme',
    '27': 'Eure',
    '28': 'Eure-et-Loir',
    '29': 'Finistère',
    '30': 'Gard',
    '31': 'Haute-Garonne',
    '32': 'Gers',
    '33': 'Gironde',
    '34': 'Hérault',
    '35': 'Ille-et-Vilaine',
    '36': 'Indre',
    '37': 'Indre-et-Loire',
    '38': 'Isère',
    '39': 'Jura',
    '40': 'Landes',
    '41': 'Loir-et-Cher',
    '42': 'Loire',
    '43': 'Haute-Loire',
    '44': 'Loire-Atlantique',
    '45': 'Loiret',
    '46': 'Lot',
    '47': 'Lot-et-Garonne',
    '48': 'Lozère',
    '49': 'Maine-et-Loire',
    '50': 'Manche',
    '51': 'Marne',
    '52': 'Haute-Marne',
    '53': 'Mayenne',
    '54': 'Meurthe-et-Moselle',
    '55': 'Meuse',
    '56': 'Morbihan',
    '57': 'Moselle',
    '58': 'Nièvre',
    '59': 'Nord',
    '60': 'Oise',
    '61': 'Orne',
    '62': 'Pas-de-Calais',
    '63': 'Puy-de-Dôme',
    '64': 'Pyrénées-Atlantiques',
    '65': 'Hautes-Pyrénées',
    '66': 'Pyrénées-Orientales',
    '67': 'Bas-Rhin',
    '68': 'Haut-Rhin',
    '69': 'Rhône',
    '70': 'Haute-Saône',
    '71': 'Saône-et-Loire',
    '72': 'Sarthe',
    '73': 'Savoie',
    '74': 'Haute-Savoie',
    '75': 'Paris',
    '76': 'Seine-Maritime',
    '77': 'Seine-et-Marne',
    '78': 'Yvelines',
    '79': 'Deux-Sèvres',
    '80': 'Somme',
    '81': 'Tarn',
    '82': 'Tarn-et-Garonne',
    '83': 'Var',
    '84': 'Vaucluse',
    '85': 'Vendée',
    '86': 'Vienne',
    '87': 'Haute-Vienne',
    '88': 'Vosges',
    '89': 'Yonne',
    '90': 'Territoire de Belfort',
    '91': 'Essonne',
    '92': 'Hauts-de-Seine',
    '93': 'Seine-Saint-Denis',
    '94': 'Val-de-Marne',
    '95': "Val-d'Oise",
    '971': 'Guadeloupe',
    '972': 'Martinique',
    '973': 'Guyane',
    '974': 'La Réunion',
    '975': 'Saint-Pierre-et-Miquelon',  # Corrigé d'une petite coquille potentielle
    '984': 'Terres australes et antarctiques françaises',
    '985': 'Mayotte',
    '986': 'Wallis-et-Futuna',
    '987': 'Polynésie française',
    '988': 'Nouvelle-Calédonie',
}

# 4. Création de la colonne NOM_DEP via le dictionnaire
df['NOM_DEP'] = df['DEP'].map(correspondance).fillna('Non répertorié')

# Aperçu du résultat
print(df[['NUM_POSTE', 'DEP', 'NOM_DEP']].head(20))

# ==========================================
# Sauvegarde de la base finale
# ==========================================
df.to_csv('data_meteo_complet.csv', index=False)
print(f'Nombre de lignes après filtrage : {len(df)}')
