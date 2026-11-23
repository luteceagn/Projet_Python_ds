# Analyse de la régularité des TGV et des conditions météorologiques

Projet réalisé dans le cadre du cours de Python pour la Data Science à l'ENSAE Paris.

## 1- Présentation du projet

Ce projet vise à étudier les liens entre les conditions météorologiques et les retards des TGV en France, entre janvier 2018 et septembre 2026.

Nous utilisons deux sources de données :

- **SNCF Open Data** : données mensuelles sur les retards, les annulations et les circulations des TGV, récupérées automatiquement depuis l'API SNCF.
- **Météo-France** : données météorologiques collectées initialement via l'API Météo-France, puis stockées sur S3 afin d'éviter de relancer une collecte particulièrement longue à chaque exécution du notebook.

L'objectif de ce projet est d'analyser la régularité des TGV en France, puis d'étudier les liens possibles entre les conditions météorologiques et les retards ferroviaires à partir des données de la SNCF et de Météo-France.

## 2- Contenu du dépôt
Le dépot contient le notebook central du projet : rapport_final.ipynb.
Dans ce notebook, dans le but de ne pas laisser de trop longs morceaux de code, nous faisons appel à plusieurs fonctions avec différentes utilités (chargement de données, création de graphique, lancement de tests de corrélation).

Ces fonctions sont répertoriées dans les fichiers .py suivants dans l'ordre de leur utilisation dans le rapport final :

- chargement_sncf.py
- chargement_gares.py
- preparation_sncf
- collecte_meteo.py
- f_stat_desc.py
- stat_desc_jointe.py
- tests_correlation_jointure.py

Le dépôt contient aussi le fichier requirements.txt qui recense l'ensemble des modules à installer pour être capable de lancer l'intégralité du projet. Enfin, le dépôt contient un .gitignore pour empêcher certains types de dossiers d'être envoyé sur le dépôt distant.

Une fois le projet lancé, votre dépôt local contiendra en plus un .env avec la clé d'identification pour l'API Météo-France.
Et plusieurs fichiers de données en format CSV (data_meteo_enrichie.csv, df_sncf_avec_meteo.csv et df_sncf_prepare.csv)

## 3- Comment lancer le projet

Pour reproduire les analyses, commencer par cloner le dépôt GitHub, puis ouvrir un terminal à la racine du projet.

### 3.1- Installation des dépendances

Installer les bibliothèques Python nécessaires :

```bash
pip install -r requirements.txt
```

### 3.2- Exécution du notebook

Ouvrir le fichier `rapport_final.ipynb` dans Jupyter ou VS Code, de préférence depuis l'environnement Onyxia, puis exécuter les cellules dans l'ordre.

## 4- Données

### 4.1- Récupération des données

#### Données SNCF

Les données de régularité des TGV ainsi que les informations géographiques des gares sont récupérées automatiquement depuis les sources SNCF Open Data.

La collecte est réalisée directement en Python à l'aide de la bibliothèque requests. Aucun téléchargement manuel des fichiers CSV n'est nécessaire.

#### Données Météo-France

Les données météorologiques ont été initialement récupérées depuis l'API Météo-France à l'aide du script collecte_meteo.py.

Cette collecte étant particulièrement longue, les données récupérées ont été stockées sur S3 afin d'éviter de relancer l'ensemble des requêtes à chaque exécution.

Le notebook charge donc directement le fichier meteo_mensuelle_2018_2026.csv depuis S3 à l'aide de la bibliothèque s3fs.

La clé API Météo-France n'est pas nécessaire pour exécuter le notebook avec les données déjà stockées sur S3. Elle est uniquement nécessaire pour relancer la collecte initiale.


#### 🔑 Configuration de l'API Météo-France

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


### 4.2- Présentation des variables
#### Variables du traffic de la SNCF
Les données SNCF renseignent la régularité des TGV à l'échelle mensuelle. Chaque observation correspond à une liaison orientée (gare de départ → gare d'arrivée) pour un mois donné.

Les principales variables utilisées dans notre analyse sont regroupées en quatre catégories.

**1. Identification des liaisons**

- `date` : mois et année de l'observation.
- `gare_depart` et `gare_arrivee` : gares de départ et d'arrivée de la liaison.
- `service` : catégorie du service ferroviaire (National ou International).
- `duree_moyenne` : durée moyenne du trajet, en minutes.

Les liaisons dont une gare se situe à l'étranger sont exclues. En revanche, les liaisons classées « International » sont conservées lorsque les deux gares sont situées en France.

**2. Circulations et annulations**

- `nb_train_prevu` : nombre de trains prévus sur la liaison pendant le mois.
- `nb_annulation` : nombre de trains annulés.
- `circulations_effectives` : nombre de trains prévus diminué du nombre d'annulations. Cette variable est calculée dans le projet.

**3. Retards des TGV**

- `nb_train_retard_arrivee` : nombre de trains considérés en retard à l'arrivée selon les critères de régularité de la SNCF.
- `nb_train_depart_retard` : nombre de trains en retard au départ.
- `retard_moyen_depart` et `retard_moyen_arrivee` : indicateurs de retard moyen, exprimés en minutes.
- `nb_train_retard_sup_15`, `nb_train_retard_sup_30` et `nb_train_retard_sup_60` : nombres de trains dont le retard dépasse respectivement 15, 30 et 60 minutes.

Nous construisons également le `taux_retard_arrivee`, obtenu en rapportant le nombre de trains en retard à l'arrivée au nombre de circulations effectives, pour les observations où ce dernier est strictement positif.

**4. Causes des retards**

La SNCF fournit six variables représentant la répartition des causes de retard :

- `prct_cause_externe` : causes externes, pouvant notamment inclure des événements météorologiques.
- `prct_cause_infra` : problèmes liés aux infrastructures.
- `prct_cause_gestion_trafic` : difficultés de gestion du trafic ferroviaire.
- `prct_cause_materiel_roulant` : problèmes liés au matériel roulant.
- `prct_cause_gestion_gare` : difficultés de gestion en gare.
- `prct_cause_prise_en_charge_voyageurs` : problèmes liés à la prise en charge des voyageurs.

Ces variables sont exprimées en pourcentage et permettent de décrire la répartition des causes de retard déclarées par la SNCF. La catégorie des causes externes ne permet toutefois pas d'isoler les seuls événements météorologiques.

Enfin, les variables `dep_depart` et `dep_arrivee` sont ajoutées lors de la préparation des données afin d'associer à chaque liaison les conditions météorologiques de ses départements de départ et d'arrivée.

#### Variables Climatologiques
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


## 5- Préparation des données et analyses

### 5.1- Préparation des données SNCF

Nous commençons par explorer les données SNCF, notamment les gares présentes, les valeurs manquantes et la cohérence des circulations.

Afin de conserver un périmètre géographique cohérent avec les données météorologiques françaises, nous excluons les liaisons dont la gare de départ ou d'arrivée se situe à l'étranger. Ce filtrage permet de passer de **12 907 à 11 357 observations**.

Les départements des gares sont ensuite identifiés à partir des informations géographiques fournies par la SNCF. Une normalisation des noms et des correspondances spécifiques permettent de rapprocher les différentes appellations des gares.

Enfin, nous construisons les indicateurs nécessaires à l'analyse de la régularité, notamment le nombre de circulations effectives et le taux de retard à l'arrivée.

### 5.2- Préparation des données météorologiques

Les données météorologiques sont nettoyées et agrégées à une échelle mensuelle et départementale afin de les rendre compatibles avec les données SNCF.

Nous construisons également des indicateurs permettant de caractériser les conditions météorologiques observées, notamment les températures, les précipitations et les phénomènes extrêmes.

### 5.3- Jointure des données SNCF et Météo-France

Les deux bases sont rapprochées à partir du mois d'observation et des départements associés aux gares.

Les informations météorologiques sont intégrées séparément pour les départements de départ et d'arrivée de chaque liaison.

### 5.4- Analyses réalisées

Notre étude comprend plusieurs étapes :

- Une analyse descriptive de la régularité des TGV et de son évolution temporelle.
- Une étude des retards, des annulations et de leurs causes déclarées.
- Une exploration des conditions météorologiques.
- Une analyse des indicateurs de régularité selon les conditions météorologiques observées.
- Des analyses de corrélation de Pearson et de Spearman afin d'étudier les associations statistiques entre les variables ferroviaires et météorologiques.

## 6- Principaux résultats
[à compléter]
