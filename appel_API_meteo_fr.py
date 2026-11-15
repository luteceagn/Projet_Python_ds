import io
import json
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------------ config
load_dotenv()
API_KEY = os.getenv("METEO_FRANCE_API_KEY")
if not API_KEY:
    raise ValueError("Clé METEO_FRANCE_API_KEY introuvable dans le .env")

HEADERS = {"apikey": API_KEY}  # pas de accept: json, le fichier est un CSV
BASE = "https://public-api.meteofrance.fr/public/DPClim/v1"

# Métropole + DROM.
DEPARTEMENTS = list(range(1, 96)) + [971, 972, 973, 974, 975, 984, 985, 986, 987, 988]

ANNEE_DEBUT = 2018
DEBUT_PERIODE = date(ANNEE_DEBUT, 1, 1)
OUTPUT_FILE = "meteo_mensuelle_2018_2026.csv"

CACHE_DIR = Path("cache_meteo")   # un CSV par station (permet de reprendre)
INFO_DIR = Path("cache_info")     # un JSON d'historique par station
CACHE_DIR.mkdir(exist_ok=True)
INFO_DIR.mkdir(exist_ok=True)

# Limite de l'API ~50 requêtes/min : on reste en dessous (~40/min)
MIN_INTERVAL = 1.5
_last_call = 0.0


# ------------------------------------------------------------------ outils API
def api_get(url, params, max_retries=6):
    """GET avec limitation de débit et nouvelle tentative sur 429."""
    global _last_call
    r = None
    for i in range(max_retries):
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=60)
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


def periode_station(sid, ouverte):
    """Renvoie (date_debut, date_fin) de la station, ou None si inconnue."""
    f = INFO_DIR / f"{sid}.json"
    if f.exists():
        info = json.loads(f.read_text(encoding="utf-8"))
    else:
        r = api_get(f"{BASE}/information-station", {"id-station": sid})
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


def commander_et_telecharger(station_id, date_deb, date_fin):
    """Commande le fichier mensuel, attend qu'il soit prêt, renvoie le texte CSV (ou None)."""
    r = api_get(
        f"{BASE}/commande-station/mensuelle",
        {
            "id-station": station_id,
            "date-deb-periode": date_deb,
            "date-fin-periode": date_fin,
        },
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
        rf = api_get(f"{BASE}/commande/fichier", {"id-cmde": id_cmde})
        if rf is None:
            return None
        if rf.status_code in (200, 201):
            return rf.content.decode("utf-8", errors="replace")
        if rf.status_code == 204:
            delai = min(delai + 1, 8)
            continue
        if rf.status_code == 404:
            print("     · pas de données sur cette année")  # normal, message discret
            return None
        print(f"     ❌ Fichier ({rf.status_code}) : {rf.text[:200]}")
        return None
    print("     ⏳ Fichier jamais prêt (timeout)")
    return None


def recuperer_station(station_id, annee_min, annee_max):
    """Une commande par année (l'API limite à 1 an par requête)."""
    aujourdhui = date.today()
    morceaux = []
    for annee in range(annee_min, annee_max + 1):
        d1 = f"{annee}-01-01T00:00:00Z"
        d2 = min(date(annee, 12, 31), aujourdhui)
        c = commander_et_telecharger(station_id, d1, f"{d2.isoformat()}T00:00:00Z")
        if c:
            morceaux.append(c)
    return morceaux


# ------------------------------------------------------------------ boucle principale
for dept in DEPARTEMENTS:
    print(f"\n================ Département {dept:02d} ================")
    r = api_get(f"{BASE}/liste-stations/quotidienne", {"id-departement": dept})
    if r is None or r.status_code != 200:
        print(f"⚠️ Stations indisponibles ({r.status_code if r is not None else '?'})")
        continue

    stations = [s for s in r.json() if s.get("postePublic")]
    print(f"👉 {len(stations)} stations publiques")

    for st in stations:
        sid = str(st.get("id", "")).zfill(8)
        if not sid.strip("0"):
            continue

        cache_file = CACHE_DIR / f"{sid}.csv"
        if cache_file.exists():
            continue  # déjà traitée lors d'un lancement précédent

        per = periode_station(sid, st.get("posteOuvert"))
        if per is None:
            print(f"   [{sid}] historique indisponible, ignorée (sera retentée au prochain lancement)")
            continue
        debut, fin = per

        # Garde la station si elle a été active au moins un jour entre 2018 et aujourd'hui
        if fin < DEBUT_PERIODE or debut > date.today():
            cache_file.write_text("", encoding="utf-8")
            continue

        annee_min = max(debut.year, ANNEE_DEBUT)
        annee_max = min(fin.year, date.today().year)
        print(f"   [{sid}] {st.get('nom', '')} ({debut} → {fin}) : {annee_min}-{annee_max}")

        morceaux = recuperer_station(sid, annee_min, annee_max)
        if not morceaux:
            cache_file.write_text("", encoding="utf-8")  # pas de données : on ne redemande pas
            continue

        dfs = []
        for m in morceaux:
            try:
                dfs.append(
                    pd.read_csv(io.StringIO(m), sep=";", decimal=",", dtype={"POSTE": str})
                )
            except Exception as e:
                print(f"     ⚠️ CSV illisible : {e}")

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df["departement"] = dept
            df["station_nom"] = st.get("nom")
            df.to_csv(cache_file, index=False, sep=";")
            print(f"     ✅ {len(df)} lignes")

# ------------------------------------------------------------------ fusion
fichiers = [f for f in CACHE_DIR.glob("*.csv") if f.stat().st_size > 0]
if not fichiers:
    raise SystemExit("❌ Aucune donnée récupérée.")

master = pd.concat(
    (pd.read_csv(f, sep=";", dtype={"POSTE": str}) for f in fichiers),
    ignore_index=True,
).drop_duplicates()

master.to_csv(OUTPUT_FILE, index=False, sep=";")
print(f"\n🎉 {OUTPUT_FILE} généré : {len(master)} lignes, {master['POSTE'].nunique()} stations")
