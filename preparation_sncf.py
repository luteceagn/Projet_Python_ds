from chargement_sncf import charger_sncf
from chargement_gares import charger_gares


# 1. Identification des gares de départ et d'arrivée

def lister_gares(df):

    # Récupérer les noms des gares sans doublons
    gares = sorted(
        set(df["gare_depart"].dropna())
        | set(df["gare_arrivee"].dropna())
    )

    return gares


# 2. Liste des destinations étrangères à exclure

gares_etrangeres = [
    "BARCELONA",
    "FRANCFORT",
    "GENEVE",
    "ITALIE",
    "LAUSANNE",
    "MADRID",
    "STUTTGART",
    "ZURICH"
]


# 3. Filtrage des liaisons françaises

def filtrer_liaisons_francaises(df):

    # Conserver uniquement les liaisons
    # dont le départ et l'arrivée sont en France
    df_filtre = df[
        (~df["gare_depart"].isin(gares_etrangeres))
        & (~df["gare_arrivee"].isin(gares_etrangeres))
        & (df["gare_depart"].notna())
        & (df["gare_arrivee"].notna())
    ].copy()

    return df_filtre


# 4. Vérifications

if __name__ == "__main__":

    # Chargement des données SNCF
    df_sncf = charger_sncf()

    # Filtrage des liaisons françaises
    df_sncf_fr = filtrer_liaisons_francaises(df_sncf)

    print("Nombre de lignes initial :", len(df_sncf))
    print("Nombre de lignes restantes :", len(df_sncf_fr))
    print("Nombre de lignes supprimées :", len(df_sncf) - len(df_sncf_fr))

    print("Nombre de gares initial :", len(lister_gares(df_sncf)))
    print("Nombre de gares après filtrage :", len(lister_gares(df_sncf_fr)))

    # Chargement des données géographiques des gares
    df_gares = charger_gares()

    # Comparaison des noms dans les deux bases
    noms_sncf = set(lister_gares(df_sncf_fr))
    noms_geo = set(
        df_gares["nom"].dropna().str.strip().str.upper()
    )

    # Identification des correspondances
    gares_trouvees = noms_sncf & noms_geo
    gares_non_trouvees = noms_sncf - noms_geo

    print("\nNombre de correspondances exactes :", len(gares_trouvees))
    print("Nombre de gares sans correspondance :", len(gares_non_trouvees))

    print("\nGares sans correspondance exacte :")

    for gare in sorted(gares_non_trouvees):
        print(gare)