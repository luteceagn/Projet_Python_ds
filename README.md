# Projet_Python_ds

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
METEO_FRANCE_API_KEY=votre_cle_api_ici```
