from datetime import date
import io
import json
import os
from pathlib import Path
import time
from dotenv import load_dotenv
import pandas as pd
import requests

# ----------------------------------- configuration -----------------------------------
BASE = "https://public-api.meteofrance.fr/public/DPClim/v1"

# Sélection des départements par défaut : Métropole + DROM / COM
DEPARTEMENTS_DEFAUT = list(range(1, 96)) + [
    971,
    972,
    973,
    974,
    975,
    984,
    985,
    986,
    987,
    988,
]

# Limite de l'API ~50 requêtes/min : on reste en dessous (~40/min), évite les erreurs
MIN_INTERVAL = 1.5
_last_call = 0.0


def _headers():
    """Charge la clé API Météo-France depuis le fichier .env et renvoie l'en-tête HTTP."""
    env_module = Path(__file__).resolve().parent / ".env"
    env_cwd = Path.cwd() / ".env"

    if env_module.exists():
        load_dotenv(env_module, override=True)
    elif env_cwd.exists():
        load_dotenv(env_cwd, override=True)
    else:
        load_dotenv(override=True)

    api_key = os.getenv("METEO_FRANCE_API_KEY")
    if not api_key:
        raise ValueError("Clé METEO_FRANCE_API_KEY introuvable dans le .env")

    return {"apikey": api_key.strip()}


# ------------------------------------- outils API -------------------------------------
def api_get(url, params, headers, max_retries=6):
    """GET avec limitation de débit et nouvelle tentative sur 429."""
    global _last_call
    r = None
    for i in range(max_retries):
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            r = requests.get(url, headers=headers, params=params, timeout=60)
        except requests.RequestException as e:
            print(f"     ⚠️ Erreur réseau : {e}")
            time.sleep(10)
            continue
        if r.status_code == 429:
            pause = 20 * (i + 1)
            print(f"     ⏳ 429, pause de {pause}s...")
            time.sleep(pause)
            continue
        return r
    return r


def _parse_date(v):
    if not v:
        return None
    d = pd.to_datetime(v, errors="coerce", utc=True)
    return None if pd.isna(d) else d.date()


def periode_station(sid, ouverte, headers, info_dir):
    """Renvoie (date_debut, date_fin) de la station, ou None si inconnue."""
    f = info_dir / f"{sid}.json"
    if f.exists():
        info = json.loads(f.read_text(encoding="utf-8"))
    else:
        r = api_get(
            f"{BASE}/information-station", {"id-station": sid}, headers
        )
        if r is None or r.status_code != 200:
            return None
        info = r.json()
        f.write_text(json.dumps(info), encoding="utf-8")

    if isinstance(info, list):
        info = info[0] if info else {}

    debut = _parse_date(info.get("dateDebut"))
    fin = _parse_date(info.get("dateFin"))
    if debut is None:
        return None

    # Station encore ouverte, sans date de fin, ou fin dans le futur : jusqu'à aujourd'hui
    if ouverte or fin is None or fin > date.today():
        fin = date.today()
    return debut, fin


def commander_et_telecharger(station_id, date_deb, date_fin, headers):
    """Commande le fichier mensuel, attend qu'il soit prêt, renvoie le texte CSV (ou None)."""
    r = api_get(
        f"{BASE}/commande-station/mensuelle",
        {
            "id-station": station_id,
            "date-deb-periode": date_deb,
            "date-fin-periode": date_fin,
        },
        headers,
    )
    if r is None or r.status_code != 202:
        code = r.status_code if r is not None else "?"
        txt = r.text[:200] if r is not None else ""
        print(f"     ❌ Commande refusée ({code}) : {txt}")
        return None

    try:
        id_cmde = r.json()["elaboreProduitAvecDemandeResponse"]["return"]
    except (KeyError, ValueError, TypeError):
        print(f"     ❌ Id de commande introuvable : {r.text[:200]}")
        return None

    # Polling : 204 = en cours de production, 201 (ou 200) = fichier prêt
    delai = 3
    for _ in range(25):
        time.sleep(delai)
        rf = api_get(f"{BASE}/commande/fichier", {"id-cmde": id_cmde}, headers)
        if rf is None:
            return None
        if rf.status_code in (200, 201):
            return rf.content.decode("utf-8", errors="replace")
        if rf.status_code == 204:
            delai = min(delai + 1, 8)
            continue
        if rf.status_code == 404:
            print("     · pas de données sur cette année")
            return None
        print(f"     ❌ Fichier ({rf.status_code}) : {rf.text[:200]}")
        return None
    print("     ⏳ Fichier jamais prêt (timeout)")
    return None


def recuperer_station(station_id, annee_min, annee_max, headers):
    """Une commande par année (l'API limite à 1 an par requête)."""
    aujourdhui = date.today()
    morceaux = []
    for annee in range(annee_min, annee_max + 1):
        d1 = f"{annee}-01-01T00:00:00Z"
        d2 = min(date(annee, 12, 31), aujourdhui)
        c = commander_et_telecharger(
            station_id, d1, f"{d2.isoformat()}T00:00:00Z", headers
        )
        if c:
            morceaux.append(c)
    return morceaux


# --------------------------------- fonction principale --------------------------------
def recuperer_meteo(
    annee_debut,
    annee_fin,
    departements=DEPARTEMENTS_DEFAUT,
    output_file=None,
    cache_dir="cache_meteo",
    info_dir="cache_info",
    sauvegarder=True,
):
    """Récupère les données climatiques mensuelles Météo-France entre `annee_debut` et `annee_fin`."""
    if annee_debut > annee_fin:
        raise ValueError(
            f"`annee_debut` ({annee_debut}) doit être inférieur ou égal à `annee_fin` ({annee_fin})."
        )

    headers = _headers()
    debut_periode = date(annee_debut, 1, 1)
    fin_periode = min(date(annee_fin, 12, 31), date.today())

    cache_path = Path(f"{cache_dir}_{annee_debut}_{annee_fin}")
    info_path = Path(info_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    info_path.mkdir(parents=True, exist_ok=True)

    if output_file is None:
        output_file = f"meteo_mensuelle_{annee_debut}_{annee_fin}.csv"

    for dept in departements:
        dept_int = int(dept)
        print(f"\n================ Département {dept_int:02d} ================")
        r = api_get(
            f"{BASE}/liste-stations/quotidienne",
            {"id-departement": dept_int},
            headers,
        )
        if r is None or r.status_code != 200:
            print(
                f"⚠️ Stations indisponibles ({r.status_code if r is not None else '?'})"
            )
            continue

        stations = [s for s in r.json() if s.get("postePublic")]
        print(f"👉 {len(stations)} stations publiques")

        for st in stations:
            sid = str(st.get("id", "")).zfill(8)
            if not sid.strip("0"):
                continue

            cache_file = cache_path / f"{sid}.csv"
            if cache_file.exists():
                continue  # déjà traitée lors d'un lancement précédent

            per = periode_station(sid, st.get("posteOuvert"), headers, info_path)
            if per is None:
                print(
                    f"   [{sid}] historique indisponible, ignorée (sera retentée au prochain lancement)"
                )
                continue
            debut, fin = per

            # Garde la station si elle a été active au moins un jour sur la période demandée
            if fin < debut_periode or debut > fin_periode:
                cache_file.write_text("", encoding="utf-8")
                continue

            annee_min = max(debut.year, annee_debut)
            annee_max = min(fin.year, fin_periode.year)
            print(
                f"   [{sid}] {st.get('nom', '')} ({debut} → {fin}) : {annee_min}-{annee_max}"
            )

            morceaux = recuperer_station(sid, annee_min, annee_max, headers)
            if not morceaux:
                cache_file.write_text("", encoding="utf-8")
                continue

            dfs = []
            for m in morceaux:
                try:
                    dfs.append(
                        pd.read_csv(
                            io.StringIO(m),
                            sep=";",
                            decimal=",",
                            dtype={"POSTE": str},
                        )
                    )
                except Exception as e:
                    print(f"     ⚠️ CSV illisible : {e}")

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                df["departement"] = dept_int
                df["station_nom"] = st.get("nom")
                df.to_csv(cache_file, index=False, sep=";")
                print(f"     ✅ {len(df)} lignes")

    # --------------------------------------- fusion ---------------------------------------
    fichiers = [f for f in cache_path.glob("*.csv") if f.stat().st_size > 0]
    if not fichiers:
        raise RuntimeError("❌ Aucune donnée récupérée.")

    master = pd.concat(
        (pd.read_csv(f, sep=";", dtype={"POSTE": str}) for f in fichiers),
        ignore_index=True,
    ).drop_duplicates()

    if sauvegarder:
        master.to_csv(output_file, index=False, sep=";")
        print(
            f"\n🎉 {output_file} généré : {len(master)} lignes, {master['POSTE'].nunique()} stations"
        )

    return master
