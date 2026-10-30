
import re
import unicodedata
import pandas as pd

from chargement_sncf import charger_sncf
from chargement_gares import charger_gares


# 1. Identification des gares

def lister_gares(df):
    return sorted(
        set(df["gare_depart"].dropna())
        | set(df["gare_arrivee"].dropna())
    )


# 2. Filtrage des liaisons françaises

gares_etrangeres = [
    "BARCELONA", "FRANCFORT", "GENEVE", "ITALIE",
    "LAUSANNE", "MADRID", "STUTTGART", "ZURICH"
]


def filtrer_liaisons_francaises(df):
    return df[
        ~df["gare_depart"].isin(gares_etrangeres)
        & ~df["gare_arrivee"].isin(gares_etrangeres)
        & df["gare_depart"].notna()
        & df["gare_arrivee"].notna()
    ].copy()


# 3. Normalisation des noms

def normaliser_nom(nom):
    nom = unicodedata.normalize("NFKD", str(nom))
    nom = "".join(
        c for c in nom if not unicodedata.combining(c)
    )

    nom = nom.upper()
    nom = re.sub(r"[-'’(),.]", " ", nom)

    # Harmonisation des abréviations
    nom = re.sub(r"\bST\b", "SAINT", nom)
    nom = re.sub(r"\bSTE\b", "SAINTE", nom)

    nom = re.sub(r"\s+", " ", nom).strip()

    return nom


# 4. Différences d'appellation vérifiées
# On conserve les noms d'origine de la base SNCF.

correspondances_noms = {
    "BELLEGARDE (AIN)": "Bellegarde-sur-Valserine",
    "DIJON VILLE": "Dijon",
    "LA ROCHELLE VILLE": "La Rochelle",
    "LE CREUSOT MONTCEAU MONTCHANIN":
        "Le Creusot - Montceau-les-Mines - Montchanin TGV",
    "MACON LOCHE": "Mâcon Loché TGV",
    "MARNE LA VALLEE": "Marne-la-Vallée Chessy",
    "MULHOUSE VILLE": "Mulhouse",
    "NICE VILLE": "Nice",
    "PARIS LYON": "Paris Gare de Lyon",
    "PARIS NORD": "Paris Gare du Nord",
    "VALENCE ALIXAN TGV": "Valence TGV Rhône-Alpes Sud"
}


# 5. Cas où seul le département est identifié
# Ne pas utiliser ces correspondances pour les coordonnées GPS.

departements_exceptions = {
    "LILLE": "59",
    "MONTPELLIER": "34",
    "PARIS VAUGIRARD": "75"
}


# 6. Extraction du département depuis le code INSEE

def extraire_departement(code_insee):
    code = str(code_insee).strip().upper()

    if not re.fullmatch(r"(?:\d{5}|2[AB]\d{3})", code):
        raise ValueError(f"Code INSEE invalide : {code}")

    # Départements et collectivités d'outre-mer
    if code.startswith(("97", "98")):
        return code[:3]

    # Corse
    if code.startswith(("2A", "2B")):
        return code[:2]

    return code[:2]


# 7. Construction de la table de correspondance

def construire_correspondances(df_sncf, df_gares):

    df_ref = df_gares.dropna(
        subset=["nom", "codeinsee"]
    ).copy()

    df_ref["codeinsee"] = (
        df_ref["codeinsee"]
        .astype("string")
        .str.strip()
        .str.zfill(5)
    )

    df_ref["nom_normalise"] = (
        df_ref["nom"].map(normaliser_nom)
    )

    correspondances = []

    for gare in lister_gares(df_sncf):

        # Cas particuliers : département connu,
        # sans identifier arbitrairement une gare précise
        if gare in departements_exceptions:
            correspondances.append({
                "gare_sncf": gare,
                "nom_geo": pd.NA,
                "codeinsee": pd.NA,
                "departement": departements_exceptions[gare],
                "methode": "Département uniquement"
            })
            continue

        # Utiliser l'appellation corrigée si nécessaire
        nom_recherche = correspondances_noms.get(gare, gare)
        nom_normalise = normaliser_nom(nom_recherche)

        candidats = df_ref[
            df_ref["nom_normalise"] == nom_normalise
        ][["nom", "codeinsee"]].drop_duplicates()

        # Vérifier qu'une correspondance existe
        if candidats.empty:
            raise ValueError(
                f"Aucune correspondance pour : {gare}"
            )

        # Vérifier qu'il n'y a pas plusieurs codes INSEE
        codes = candidats["codeinsee"].unique()

        if len(codes) != 1:
            raise ValueError(
                f"Correspondance ambiguë pour {gare} : {codes}"
            )

        code_insee = codes[0]

        correspondances.append({
            "gare_sncf": gare,
            "nom_geo": candidats.iloc[0]["nom"],
            "codeinsee": code_insee,
            "departement": extraire_departement(code_insee),
            "methode": (
                "Appellation corrigée"
                if gare in correspondances_noms
                else "Normalisation automatique"
            )
        })

    table = pd.DataFrame(correspondances)

    # Contrôles
    assert table["gare_sncf"].is_unique
    assert table["departement"].notna().all()
    assert len(table) == len(lister_gares(df_sncf))

    return table


# 8. Ajout des départements dans la base SNCF

def ajouter_departements(df, table):

    # Première jointure : département de départ
    df_final = df.merge(
        table[["gare_sncf", "departement"]].rename(
            columns={
                "gare_sncf": "gare_depart",
                "departement": "dep_depart"
            }
        ),
        on="gare_depart",
        how="left",
        validate="many_to_one"
    )

    # Deuxième jointure : département d'arrivée
    df_final = df_final.merge(
        table[["gare_sncf", "departement"]].rename(
            columns={
                "gare_sncf": "gare_arrivee",
                "departement": "dep_arrivee"
            }
        ),
        on="gare_arrivee",
        how="left",
        validate="many_to_one"
    )

    # Vérifier qu'aucune ligne n'a été perdue ou dupliquée
    assert len(df_final) == len(df)

    # Vérifier les départements manquants
    assert df_final["dep_depart"].notna().all()
    assert df_final["dep_arrivee"].notna().all()

    return df_final


# 9. Fonction principale, réutilisable dans le notebook

def preparer_sncf():

    df_sncf = charger_sncf()

    df_sncf_fr = filtrer_liaisons_francaises(df_sncf)

    df_gares = charger_gares()

    table = construire_correspondances(
        df_sncf_fr, df_gares
    )

    df_final = ajouter_departements(
        df_sncf_fr, table
    )

    return df_final


# 10. Vérification lors de l'exécution du fichier

if __name__ == "__main__":

    df_sncf = charger_sncf()
    df_sncf_fr = filtrer_liaisons_francaises(df_sncf)
    df_gares = charger_gares()

    table = construire_correspondances(
        df_sncf_fr, df_gares
    )

    df_final = ajouter_departements(
        df_sncf_fr, table
    )

    print("Observations initiales :", len(df_sncf))
    print("Observations conservées :", len(df_final))

    print("\nNombre de gares :", len(table))

    print("\n--- Table des correspondances ---")
    print(table.to_string(index=False))

    print("\n--- Départements manquants ---")
    print(df_final[
        ["dep_depart", "dep_arrivee"]
    ].isna().sum())

    print("\n--- Aperçu de la base finale ---")
    print(df_final[
        ["gare_depart", "gare_arrivee",
         "dep_depart", "dep_arrivee"]
    ].head(10))

