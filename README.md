# ModelAI - Projet de Trading Algorithmique

ModelAI est un projet de trading algorithmique qui combine l'utilisation de modèles d'apprentissage automatique pour les prévisions financières et une interface utilisateur via Telegram pour le contrôle des opérations. Ce projet a été développé à l'aide de **ChatGPT** et **Mistral**, avec une architecture modulaire pour l'intégration et l'amélioration continues.

---

## 🚀 Fonctionnalités principales

- **Prédictions de marché** basées sur des modèles LSTM et XGBoost.
- **Analyse des sentiments** des actualités textuelles pour intégrer des signaux externes dans les décisions.
- **Gestion du trading en temps réel** via une interface Telegram.
- **Arborescence organisée** pour faciliter la maintenance et l'amélioration.

---

## 📂 Arborescence des fichiers

```plaintext
ModelAI/
│
├── config/
├── dataDirect/          # Données brutes
├── dataTrained/         # Modèles entraînés et pré-traitements
├── models/              # Fichiers de modèles et entraînement
│   ├── entrainement/    # Scripts pour l'entraînement des modèles
│   ├── telegramControler/
│   └── ...
├── MT5/                 # Intégration avec MetaTrader5
├── outils_historique/   # Historique et outils d'analyse
├── trade_info/          # Gestion des informations de trading
└── templates/
    ├── historique/      # Modèles d'historique

🛠️ Installation
Prérequis

    Python 3.8+
    MetaTrader5
    Bibliothèques Python nécessaires (listées dans requirements.txt).

Étapes

    Clonez le projet :

git clone <repository_url>
cd ModelAI

Installez les dépendances :

pip install -r requirements.txt

Configurez les paramètres nécessaires dans le répertoire config/.

(Optionnel) Démarrez le serveur Flask pour la communication avec les scripts MT5 :

    python server.py

📖 Utilisation
1. Prédictions et Entraînement

Les modèles sont entraînés via le répertoire models/entrainement. Voici les étapes clés :

    Pré-traitement des données :

python preprocess.py

Entraînement du modèle LSTM :

python train_lstm.py

Entraînement XGBoost :

    python train_xgboost.py

2. Contrôle via Telegram

Configurez et exécutez le contrôleur Telegram dans telegramControler/ :

    Lancer le bot :

    python TelegramControler.py

    Commandes disponibles :
        /start : Affiche les commandes disponibles.
        /set_volume <volume> : Définit le volume des transactions.
        /blockTrading : Bloque les exécutions de trading.
        /unlockAll : Débloque les exécutions de trading.
        /stats : Affiche les statistiques (TP/SL moyens, etc.).

3. Intégration avec MT5

Le script MTCode_AiPredictionFlask.mq5 est utilisé pour intégrer le trading algorithmique avec MetaTrader5.
🧩 Points clés du code

    MistralAI_model.py
        Entraîne des modèles avec des données boursières et textuelles.
        Génère des prédictions basées sur des indicateurs techniques et des analyses de sentiments.

    TelegramControler.py
        Fournit un contrôle en temps réel via des commandes Telegram.
        Utilise des API REST pour gérer les actions du bot.

    MTCode_AiPredictionFlask.mq5
        S'occupe de l'intégration avec MetaTrader5.
        Permet la récupération et l'envoi de données de trading au serveur Flask.

🖼️ Aperçu des scripts
Exemple de commande Telegram

    /set_volume 1 : Définit le volume de transaction à 1.

Exemple de prédictions LSTM

Les données d'entrée sont pré-traitées avec des indicateurs techniques et des analyses de sentiments pour maximiser la précision.
🛡️ Contribution

Les contributions sont les bienvenues ! Veuillez créer une issue ou soumettre une pull request.
📄 Licence

Ce projet est sous licence MIT.


Ce fichier README est conçu pour être clair et accueillant, tout en offrant des détails précis sur l'installation, l'utilisation et l'arborescence des fichiers. Si vous avez d'autres exigences spécifiques, n'hésitez pas à me les indiquer !
