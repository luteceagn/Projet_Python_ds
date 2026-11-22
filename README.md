
# Analyse de la régularité des TGV et des conditions météorologiques

Projet réalisé dans le cadre du cours de Python pour la Data Science à l'ENSAE Paris.

## Présentation du projet

Ce projet vise à étudier les liens entre les conditions météorologiques et les retards des TGV en France, entre janvier 2018 et septembre 2026.

Nous utilisons deux sources de données :
- **SNCF Open Data** : données mensuelles sur les retards, les annulations et la circulation des TGV.
- **Météo-France** : données météorologiques mensuelles récupérées via une API.

L'objectif de ce projet est d'analyser la régularité des TGV en France, puis d'étudier les liens possibles entre les conditions météorologiques et les retards ferroviaires à partir des données de la SNCF et de Météo-France.


## Installation et exécution du projet

Pour reproduire les analyses, commencer par cloner le dépôt GitHub, puis ouvrir un terminal à la racine du projet.

### 1. Installation des dépendances

Installer les bibliothèques Python nécessaires :

```bash
pip install -r requirements.txt
```


### 2. Exécution du notebook

Ouvrir le fichier `rapport_final.ipynb` dans Jupyter ou VS Code, puis exécuter les cellules dans l'ordre.

Le notebook utilise les fonctions définies dans les fichiers Python du projet pour récupérer et préparer automatiquement les données.

Les données SNCF, comprenant la régularité des TGV et les informations géographiques des gares, sont récupérées directement depuis les API SNCF, sans téléchargement manuel.

Les traitements et les analyses sont ensuite réalisés dans le notebook.

## 🔑 Configuration de l'API Météo-France

Pour exécuter le script d'extraction des données climatologiques, vous devez configurer votre clé d'accès personnelle.

### 1. Obtenir une clé d'API
1. Créez un compte sur le [Portail des API Météo-France](https://portail-api.meteofrance.fr).
2. Abonnez vous à l'API **Données Climatologiques** (DPClim).
3. Récupérez votre clé d'accès (*API Key*) depuis votre espace personnel.

### 2. Configurer le fichier `.env`
À la racine du projet, créez un fichier nommé **`.env`** (ce fichier contient vos identifiants secrets et ne doit jamais être partagé).

Ajoutez-y la ligne suivante en remplaçant la valeur par votre propre clé :
```env
METEO_FRANCE_API_KEY=votre_cle_api_ici
```


## Démarche et préparation des données

### Données SNCF

Les données de régularité des TGV sont récupérées automatiquement depuis l'API SNCF à l'aide de la bibliothèque `requests`, puis chargées dans un DataFrame `pandas`. La période retenue s'étend de janvier 2018 à septembre 2026.

Nous commençons par explorer la base, notamment les gares présentes et les valeurs manquantes. Pour conserver un périmètre géographique cohérent avec les données météorologiques françaises, nous excluons les liaisons dont la gare de départ ou d'arrivée se situe à l'étranger.

Cette étape permet de passer de 12 907 à 11 357 observations.

### Identification des départements

La base de régularité SNCF ne contient pas directement les départements des gares. Or, cette information est nécessaire pour rapprocher les données ferroviaires et météorologiques.

Nous récupérons donc un second jeu de données depuis l'API SNCF, contenant les noms des gares et leurs codes INSEE.

Les appellations étant parfois différentes entre les deux bases, nous procédons à une normalisation des noms (accents, ponctuation et abréviations), puis utilisons des correspondances spécifiques pour les cas qui ne peuvent pas être rapprochés automatiquement.

Pour trois cas particuliers, nous attribuons directement le département sans supposer une correspondance avec une gare précise.

Cette méthode permet d'identifier les départements des 51 gares françaises et d'ajouter deux colonnes à notre base : `dep_depart` et `dep_arrivee`, sans perdre d'observations.

### Analyse de la régularité des TGV

Nous construisons ensuite des indicateurs de régularité, notamment le taux de retard à l'arrivée, calculé à partir du nombre de trains en retard et du nombre de circulations effectives.

Les observations pour lesquelles ce taux ne peut pas être calculé sont écartées de son calcul, sans être supprimées de la base initiale.

Nous réalisons enfin une analyse descriptive et des visualisations pour étudier la régularité des TGV avant d'examiner ses liens possibles avec les conditions météorologiques.

## Présentation des variables
### Variables Climatologiques
Pour expliquer des retards ou des annulations de trains, nous avons ciblé les variables
météo qui impactent directement l'infrastructure ferroviaire (rails, caténaires, signalisation).

En faisant la moyenne mensuelle par département, les indicateurs les plus pertinents à retenir
se répartissent en quatre grands axes :

1. Le gel et la neige (Perturbations hivernales) :
Le froid perturbe l'alimentation électrique (givre sur les caténaires) et bloque les aiguillages.
- NBJGELEE : Nombre de jours de gel (essentiel pour voir l'impact du froid matinal).
- NBJSOLNG : Nombre de jours avec sol enneigé. (supprimé à cause d'un trop grand % de NA)
- TN : Moyenne mensuelle des températures minimales quotidiennes.

2. Les fortes précipitations, brouillard et orages (Risques d'inondation et de visibilité)
L'eau en excès provoque des affaissements de voies, des glissements de terrain
ou des limitations de vitesse par précaution.
- RR : Cumul mensuel des précipitations (en mm).
- NBJRR10 : Nombre de jours de fortes pluies (>=10mm).
- NBJORAG : Nombre de jours d'orage (souvent liés aux coupures de courant et pannes de signalisation).
- NBJBROU : Nombre de jours de brouillard dans le mois

3. Le vent violent (Chutes d'arbres et caténaires)
Le vent fort est l'une des causes majeures d'incidents externes sur le réseau SNCF.
- NBJFF16 : Nombre de jours avec des vents forts.

4. La chaleur extrême (Dilatation thermique)
En cas de canicule, les rails se dilatent, ce qui oblige la SNCF à imposer des
ralentissements de sécurité (LTV - Limitations Temporaires de Vitesse).
- TX : Moyenne mensuelle des températures maximales quotidiennes.
- NBJTX30 : Nombre de jours de forte chaleur. (Température >= 30 degrés)
- TAMPLIM : Amplitude thermique moyenne du mois
*Remarque*: Chaque variable de comptage (NB..) a été transformé en % de jours du mois affecté par le phénomène
