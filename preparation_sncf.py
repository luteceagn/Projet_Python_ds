from chargement_sncf import charger_sncf


# 1. Identification des gares de départ et d'arrivée

def lister_gares(df):
    # Récupérer les gares sans doublons
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


# 4. Vérification du filtrage

if __name__ == "__main__":

    # Charger les données SNCF
    df_sncf = charger_sncf()

    # Appliquer le filtrage
    df_sncf_fr = filtrer_liaisons_francaises(df_sncf)

    # Afficher les résultats
    print("Nombre de lignes initial :", len(df_sncf))
    print("Nombre de lignes restantes :", len(df_sncf_fr))
    print("Nombre de lignes supprimées :", len(df_sncf) - len(df_sncf_fr))

    print("Nombre de gares initial :", len(lister_gares(df_sncf)))
    print("Nombre de gares après filtrage :", len(lister_gares(df_sncf_fr)))

    # Vérifier les gares conservées
    print("\nListe des gares conservées :")

    for gare in lister_gares(df_sncf_fr):
        print(gare)