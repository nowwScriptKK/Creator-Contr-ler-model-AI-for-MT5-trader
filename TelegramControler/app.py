from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackContext
import requests
import pyautogui  # Pour les captures d'écran

# Remplacez par le token de votre bot Telegram
TELEGRAM_TOKEN = "8"

# Structure de données pour les utilisateurs et leurs permissions
USER_PERMISSIONS = {
    "7711267830": ["/start", "/set_volume", "/stats", "/screen", "/unlockAll", "/blockTrading", "/CloseBlockAll"],
    "7897893812": ["/start"]
}

# Fonction pour vérifier l'autorisation
def is_authorized(user_id: int, command: str) -> bool:
    user_id_str = str(user_id)
    return user_id_str in USER_PERMISSIONS and command in USER_PERMISSIONS[user_id_str]

# Fonction pour traiter la commande /set_volume
async def set_volume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id, "/set_volume"):
        await update.message.reply_text("Vous n'êtes pas autorisé à utiliser cette commande.")
        return

    try:
        # Vérifier si un argument (volume) a été fourni
        if len(context.args) != 1:
            await update.message.reply_text("Veuillez spécifier un volume. Exemple : /set_volume 1")
            return

        # Récupérer le volume depuis les arguments
        volume = context.args[0]

        # Valider que le volume est un nombre (entier ou flottant)
        if not volume.replace('.', '', 1).isdigit():
            await update.message.reply_text("Le volume doit être un nombre. Exemple : /set_volume 1")
            return

        # Convertir le volume en float
        volume = float(volume)

        # Vérifier que le volume est dans la plage autorisée
        if not 0.01 <= volume <= 100:
            await update.message.reply_text("Le volume doit être compris entre 0.01 et 100.")
            return

        # Envoyer la requête POST avec le volume en JSON
        url = "http://127.0.0.1:5000/set_volume"
        data = {"volume": volume}
        response = requests.post(url, json=data)

        # Vérifier la réponse du serveur
        if response.status_code == 200:
            await update.message.reply_text(f"Volume réglé à {volume}.")
        else:
            await update.message.reply_text("Erreur lors de l'envoi de la requête au serveur.")

    except Exception as e:
        # Gestion des erreurs
        await update.message.reply_text(f"Une erreur s'est produite : {str(e)}")

# Fonction appelée lorsqu'on envoie la commande /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    print(f"Commande /start reçue de l'utilisateur ID: {user_id}")

    if not is_authorized(user_id, "/start"):
        await update.message.reply_text("Vous n'êtes pas autorisé à utiliser ce bot.")
        print("Utilisateur non autorisé bloqué.")
        return

    # Message d'accueil avec les commandes disponibles
    welcome_message = (
        "Bienvenue sur le bot de contrôle ! \n\nIl faut demander les droits d'accès.\n\nVoici les fonctionnalités disponibles :\n\n"
        "/CloseBlockAll - Ferme toutes les positions et bloque les trades\n"
        "/blockTrading - Bloque l'exécution du trading\n"
        "/unlockAll - Débloque l'exécution du trading\n"
        "/screen - Prend une capture d'écran et l'envoie\n"
        "/stats - Affiche la moyenne des prediction TP/SL etc\n"
        "/set_volume X - Détermine le volume pour les trades\n"
        "/start - Affiche ce message d'aide\n"
    )
    await update.message.reply_text(welcome_message)

# Fonction pour envoyer une requête GET à /CloseBlockAll
async def CloseBlockAll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id, "/CloseBlockAll"):
        await update.message.reply_text("Vous n'êtes pas autorisé à utiliser cette commande.")
        return

    try:
        # Effectuer la requête GET avec le paramètre delete
        url = f"http://127.0.0.1:5000/close_all_positions?delete=True"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data["status"] == 404:
                await update.message.reply_text("Les positions ont été fermées et l'état a été réinitialisé.")
            else:
                await update.message.reply_text("Les positions ont été fermées avec succès.")
        else:
            await update.message.reply_text("Erreur lors de la fermeture des positions.")
    except Exception as e:
        await update.message.reply_text(f"Erreur : {e}")

# Fonction appelée lorsqu'on envoie la commande /blockTrading
async def block_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    print(f"Commande /blockTrading reçue de l'utilisateur ID: {user_id}")

    if not is_authorized(user_id, "/blockTrading"):
        await update.message.reply_text("Vous n'êtes pas autorisé à utiliser cette commande.")
        print("Utilisateur non autorisé bloqué.")
        return

    try:
        # Effectuer une requête POST au serveur Flask pour bloquer
        response = requests.post("http://127.0.0.1:5000/blockall")
        if response.status_code == 200:
            await update.message.reply_text("Exécution bloquée avec succès.")
        else:
            await update.message.reply_text("Erreur lors de la tentative de blocage.")
    except Exception as e:
        await update.message.reply_text(f"Erreur : {e}")

# Fonction appelée lorsqu'on envoie la commande /unlockAll
async def unlock_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    print(f"Commande /unlockAll reçue de l'utilisateur ID: {user_id}")

    if not is_authorized(user_id, "/unlockAll"):
        await update.message.reply_text("Vous n'êtes pas autorisé à utiliser cette commande.")
        print("Utilisateur non autorisé bloqué.")
        return

    try:
        # Effectuer une requête POST au serveur Flask pour débloquer
        response = requests.post("http://127.0.0.1:5000/unblock")
        if response.status_code == 200:
            await update.message.reply_text("Exécution débloquée avec succès.")
        else:
            await update.message.reply_text("Erreur lors de la tentative de déblocage.")
    except Exception as e:
        await update.message.reply_text(f"Erreur : {e}")

# Fonction appelée lorsqu'on envoie la commande /screen
async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    print(f"Commande /screen reçue de l'utilisateur ID: {user_id}")

    if not is_authorized(user_id, "/screen"):
        await update.message.reply_text("Vous n'êtes pas autorisé à utiliser cette commande.")
        print("Utilisateur non autorisé bloqué.")
        return

    try:
        # Prendre une capture d'écran
        screenshot = pyautogui.screenshot()
        screenshot_path = "screenshot.png"
        screenshot.save(screenshot_path)

        # Envoyer la capture d'écran à l'utilisateur
        await update.message.reply_photo(photo=open(screenshot_path, "rb"))
        print("Capture d'écran envoyée avec succès.")
    except Exception as e:
        await update.message.reply_text(f"Erreur lors de la capture d'écran : {e}")
        print(f"Erreur lors de la capture d'écran : {e}")

# Fonction pour récupérer et calculer les moyennes des statistiques
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id, "/stats"):
        await update.message.reply_text("Vous n'êtes pas autorisé à utiliser cette commande.")
        return

    try:
        # Récupérer les données JSON depuis l'API
        response = requests.get("http://127.0.0.1:5000/predictions_data")
        data = response.json()

        # Dictionnaire pour stocker les sommes et les comptes des symboles
        stats = {}

        for symbol, predictions in data.items():
            total_tp = total_sl = total_buy_price = total_order_type = 0
            count = 0

            # Calculer les moyennes pour chaque symbole
            for entry in predictions:
                total_tp += entry['TP']
                total_sl += entry['SL']
                total_buy_price += entry['buy_price']
                total_order_type += entry['order_type']
                count += 1

            # Moyenne pour chaque symbole
            avg_tp = total_tp / count
            avg_sl = total_sl / count
            avg_buy_price = total_buy_price / count
            avg_order_type = total_order_type / count

            # Stocker les résultats dans le dictionnaire
            stats[symbol] = {
                'avg_tp': avg_tp,
                'avg_sl': avg_sl,
                'avg_buy_price': avg_buy_price,
                'avg_order_type': avg_order_type
            }

        # Créer un message à afficher avec les résultats
        stats_message = "Statistiques des symboles :\n\n"
        for symbol, values in stats.items():
            stats_message += (
                f"{symbol} :\n"
                f"  Moyenne TP : {values['avg_tp']:.2f}\n"
                f"  Moyenne SL : {values['avg_sl']:.2f}\n"
                f"  Moyenne Buy Price : {values['avg_buy_price']:.2f}\n"
                f"  Moyenne Order Type (1=Buy, 0=Sell) : {values['avg_order_type']:.2f}\n\n"
            )

        # Envoyer le message au utilisateur
        await update.message.reply_text(stats_message)

    except Exception as e:
        await update.message.reply_text(f"Erreur lors de la récupération des données : {e}")

def main():
    # Initialiser l'application Telegram avec le token
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Ajouter les commandes au dispatcher
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("blockTrading", block_trading))
    application.add_handler(CommandHandler("unlockAll", unlock_all))
    application.add_handler(CommandHandler("screen", screen))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("set_volume", set_volume))
    application.add_handler(CommandHandler("CloseBlockAll", CloseBlockAll))

    # Lancer le bot
    application.run_polling()

if __name__ == "__main__":
    main()
