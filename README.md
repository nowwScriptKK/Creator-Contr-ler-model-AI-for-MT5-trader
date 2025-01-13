# ModelAI - Projet de Trading Algorithmique

ModelAI est un projet de trading algorithmique qui combine l'utilisation de modèles d'apprentissage automatique pour les prévisions financières et une interface utilisateur via Telegram pour le contrôle des opérations. Il permet, grâce à l'API Flask + le code MetaTrader 5 de faire de la prise de position de trade.

Il y a aussi une interface web permettant d'afficher les prédictions du model graphiquement.

# Aucunes maintenance, aucuns support d'installation pour ce projet.

---

## 🚀 Fonctionnalités principales

- **Prédictions de marché** basées sur des modèles LSTM et XGBoost.
- **Analyse des sentiments** des actualités textuelles pour intégrer des signaux externes dans les décisions.(API de la FED, il est possible de rajouté ChatGPT API ou d'utiliser le model de Mistral en local pour avoir une analyse textuel)
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
```

🛠️ Installation
Prérequis

    Python 3.8+
    MetaTrader5
    vs_BuildTools
    Bibliothèques Python nécessaires (listées dans requirements.txt).

Étapes

    Clonez le projet :
```
git clone <repository_url>
cd ModelAI
```


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

Les contributions sont les bienvenues, mais bon courage ! 
Veuillez créer une issue ou soumettre une pull request.

