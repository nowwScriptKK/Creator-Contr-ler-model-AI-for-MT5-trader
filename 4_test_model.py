import os
import re
import pandas as pd
import numpy as np
import shutil
from tensorflow.keras.models import load_model
import joblib
import json
from datetime import datetime, timedelta
import time
from flask import Flask, request, jsonify, render_template, send_from_directory, session
import threading
import uuid
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from flask_cors import CORS
import signal
import sys
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO)

# Télécharger les ressources NLTK
nltk.download('vader_lexicon')

# Répertoires
DATA_DIRECT_DIR = os.path.join(os.getcwd(), "config", "dataDirect")
DATA_TRAINED_DIR = os.path.join(os.getcwd(), "config", "dataTrained")
MODEL_DIR = os.path.join(os.getcwd(), "config", "models")
SYMBOL_INFO_FILE = os.path.join(os.getcwd(), "config", "symbols_info.json")
PERFORMANCE_TIME_FILE = os.path.join(os.getcwd(), "config", "performance_time.json")
TRADE_INFO_FILE = os.path.join(os.getcwd(), "config", "trade_info.json")
PERFORMANCE_SUMMARY_FILE = os.path.join(os.getcwd(), "config", "performance_summary.json")

# Structure de configuration
config = {
    "tp_pips": 7.8,
    "sl_pips": 3.9,
    "volume": 1,
    "min_profit_potential": 1.7,
    "model_params": {
        "lstm_units": 50,
        "epochs": 100,
        "batch_size": 32,
        "patience": 10
    },
    "status": 1,
    "blocked_all": 0,
    "blocked_symbols": {}
}

# Charger les modèles sauvegardés
def load_saved_models(model_dir):
    lstm_model_path = os.path.join(model_dir, "lstm_model.h5")
    xgb_model_path = os.path.join(model_dir, "xgb_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"Le modèle LSTM n'existe pas à l'emplacement {lstm_model_path}.")
    if not os.path.exists(xgb_model_path):
        raise FileNotFoundError(f"Le modèle XGBoost n'existe pas à l'emplacement {xgb_model_path}.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Le scaler n'existe pas à l'emplacement {scaler_path}.")

    LSTM_MODEL = load_model(lstm_model_path)
    XGB_MODEL = joblib.load(xgb_model_path)
    SCALER = joblib.load(scaler_path)

    return LSTM_MODEL, XGB_MODEL, SCALER

LSTM_MODEL, XGB_MODEL, SCALER = load_saved_models(MODEL_DIR)

# Charger les informations des symboles
def load_symbol_info(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Le fichier JSON {json_file} n'existe pas.")
    if os.path.getsize(json_file) == 0:
        raise ValueError(f"Le fichier JSON {json_file} est vide.")

    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur de format dans le fichier JSON {json_file}: {e}")

# Calculer TP et SL
def calculate_tp_sl(symbol_info, direction, entry_price, tp_pips, sl_pips):
    decimals_per_pip = symbol_info['decimals_per_pip']
    tp_offset = tp_pips / (10 ** decimals_per_pip)
    sl_offset = sl_pips / (10 ** decimals_per_pip)

    if direction == "buy":
        tp = entry_price + tp_offset
        sl = entry_price - sl_offset
    else:  # "sell"
        tp = entry_price - tp_offset
        sl = entry_price + sl_offset

    return tp, sl

# Fonction pour extraire le symbole et la temporalité du nom de fichier
def extract_symbol_and_temporality(filename):
    match = re.match(r"(.+?)_([M]\d+|H1)(?:_chaine)?\.csv", filename)
    if match:
        symbol = match.group(1)
        temporality = match.group(2)
        return symbol, temporality
    else:
        raise ValueError(f"Le nom de fichier {filename} ne correspond pas au format attendu.")

# Fonction pour charger les données CSV
def load_csv_data(filepath):
    return pd.read_csv(filepath)

# Fonction pour entraîner le modèle
def train_model(data, symbol, temporality, config):
    print(f"Entraînement du modèle pour {symbol} avec temporalité {temporality}")

    # Prétraitement des données
    label_encoder_symbol = LabelEncoder()
    label_encoder_temporalite = LabelEncoder()
    data['symbol'] = label_encoder_symbol.fit_transform([symbol] * len(data))
    data['temporalite'] = label_encoder_temporalite.fit_transform([temporality] * len(data))
    data['sentiment'] = data['sentiment'].fillna(0.0)  # Mettre 0 par défaut pour les sentiments

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume', 'sentiment', 'symbol', 'temporalite']])

    # Préparer les données pour l'entraînement
    X, y = [], []
    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i])
        y.append(scaled_data[i, 3])  # Utiliser le prix de clôture comme cible
    X, y = np.array(X), np.array(y)

    # Créer le modèle LSTM
    model = Sequential()
    model.add(LSTM(units=config["model_params"]["lstm_units"], return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=config["model_params"]["lstm_units"]))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entraîner le modèle
    early_stopping = EarlyStopping(monitor='val_loss', patience=config["model_params"]["patience"], restore_best_weights=True)
    model.fit(X, y, epochs=config["model_params"]["epochs"], batch_size=config["model_params"]["batch_size"], validation_split=0.2, callbacks=[early_stopping])

    # Sauvegarder le modèle et le scaler
    model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print(f"Entraînement terminé pour {symbol} avec temporalité {temporality}")
    return True

# Fonction pour déplacer le fichier après entraînement
def move_trained_file(filepath, destination_dir):
    shutil.move(filepath, destination_dir)
    print(f"Fichier {filepath} déplacé vers {destination_dir}")

# Fonction de filtrage des trades
def filter_trades(predictions, tp_pips, sl_pips, min_profit_potential):
    filtered_predictions = []
    for prediction in predictions:
        entry_price = prediction['entry_price']
        direction = prediction['direction']
        tp, sl = calculate_tp_sl(prediction['symbol_info'], direction, entry_price, tp_pips, sl_pips)
        profit_potential = abs(tp - entry_price) / abs(sl - entry_price)
        if profit_potential >= min_profit_potential:
            filtered_predictions.append(prediction)
    return filtered_predictions

def predict_future_candles(data, symbol, temporality, config):
    symbol_info = load_symbol_info(SYMBOL_INFO_FILE)
    symbol_data = next((item for item in symbol_info['symbols'] if item['symbol'] == symbol), None)
    if not symbol_data:
        raise ValueError(f"Le symbole {symbol} n'est pas défini dans le fichier JSON.")

    # Préparer les données pour la prédiction
    input_data = np.random.rand(len(data), 10, 8)  # Ajustez la dimension ici pour inclure 8 features
    predictions_lstm = LSTM_MODEL.predict(input_data)

    tp_count = 0
    sl_count = 0
    predictions = []
    deposit = 200  # Dépôt initial
    leverage = 1  # Leverage de 1

    for i, prediction in enumerate(predictions_lstm):
        entry_price = data['close'].iloc[i]  # Utilisez le prix de clôture comme prix d'entrée
        direction = "buy" if prediction[0] > 0.5 else "sell"
        order_type = 1 if direction == "buy" else 0  # Conversion en 1 pour "buy" et 0 pour "sell"
        tp, sl = calculate_tp_sl(symbol_data, direction, entry_price, config["tp_pips"], config["sl_pips"])

        # Vérifiez si TP ou SL est atteint
        if direction == "buy":
            if data['high'].iloc[i+1:].max() >= tp:
                tp_count += 1
                deposit += (tp - entry_price) * leverage
            elif data['low'].iloc[i+1:].min() <= sl:
                sl_count += 1
                deposit -= (entry_price - sl) * leverage
        else:  # "sell"
            if data['low'].iloc[i+1:].min() <= tp:
                tp_count += 1
                deposit += (entry_price - tp) * leverage
            elif data['high'].iloc[i+1:].max() >= sl:
                sl_count += 1
                deposit -= (sl - entry_price) * leverage

        predictions.append({
            "symbol": symbol,
            "symbol_info": symbol_data,
            "direction": direction,
            "order_type": order_type,  # Ajout de order_type
            "entry_price": entry_price,
            "take_profit": tp,
            "stop_loss": sl,
            "deposit": deposit
        })

    # Filtrer les trades intéressants
    filtered_predictions = filter_trades(predictions, config["tp_pips"], config["sl_pips"], config["min_profit_potential"])

    return filtered_predictions, tp_count, sl_count, deposit

# Fonction pour afficher les bougies
def display_candles(data, temporality):
    print(f"Affichage des bougies pour la temporalité {temporality}:")
    for index, row in data.iterrows():
        print(f"{row['datetime']} - Open: {row['open']}, High: {row['high']}, Low: {row['low']}, Close: {row['close']}, Volume: {row['volume']}")

# Fonction pour enregistrer les performances de temps
def save_performance_time(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Fonction pour enregistrer les informations des trades
def save_trade_info(filepath, data):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
            # Ensure existing_data is a list
            if isinstance(existing_data, dict):
                existing_data = [existing_data]
    else:
        existing_data = []

    existing_data.extend(data)

    with open(filepath, 'w') as f:
        json.dump(existing_data, f, indent=4)

# Fonction pour enregistrer les moyennes des performances
def save_performance_summary(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Fonction pour obtenir la dernière bougie de la bonne temporalité
def get_last_candle(filepath, temporality):
    if not os.path.exists(filepath):
        return None
    data = load_csv_data(filepath)
    if data.empty:
        return None
    return data.iloc[-1]

# Fonction pour créer ou mettre à jour les fichiers CSV
def update_csv_file(filepath, data):
    if not os.path.exists(filepath):
        data.to_csv(filepath, index=False)
    else:
        try:
            existing_data = pd.read_csv(filepath)
            combined_data = pd.concat([existing_data, data]).drop_duplicates().reset_index(drop=True)
            combined_data.to_csv(filepath, index=False)
        except pd.errors.EmptyDataError:
            # Si le fichier est vide, écrire les nouvelles données directement
            data.to_csv(filepath, index=False)

# Fonction pour traiter les fichiers
def process_files(config):
    performance_time_data = []
    trade_info_data = []
    performance_summary_data = {
        "average_tp_count": 0,
        "average_sl_count": 0,
        "potential_profitability": 0,
        "total_trades": 0,
        "successful_trades": 0,
        "average_profit_per_trade": 0
    }

    for filename in os.listdir(DATA_DIRECT_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(DATA_DIRECT_DIR, filename)
            try:
                print(f"Traitement du fichier {filename}...")
                symbol, temporality = extract_symbol_and_temporality(filename)
                print(f"Symbole: {symbol}, Temporalité: {temporality}")
                data = load_csv_data(filepath)
                print(f"Données chargées: {len(data)} lignes")

                # Vérifier si le fichier a au moins 50 lignes pour M1 ou 5 lignes pour les autres temporalités
                if (temporality == "M1" and len(data) < 50) or (temporality != "M1" and len(data) < 5):
                    print(f"Le fichier {filename} a moins de lignes requises. Ignoré.")
                    continue

                display_candles(data, temporality)

                start_time = time.time()
                if train_model(data, symbol, temporality, config):
                    move_trained_file(filepath, DATA_TRAINED_DIR)
                    predictions, tp_count, sl_count, final_deposit = predict_future_candles(data, symbol, temporality, config)
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    performance_time_data.append({
                        "symbol": symbol,
                        "temporality": temporality,
                        "elapsed_time": elapsed_time
                    })

                    trade_info_data.extend(predictions)

                    print(f"Prédictions pour {symbol} ({temporality}):")
                    for pred in predictions:
                        print(pred)
                    print(f"Nombre de TP atteints: {tp_count}, Nombre de SL atteints: {sl_count}")
                    print(f"Dépôt final: {final_deposit}€")

                    performance_summary_data["average_tp_count"] += tp_count
                    performance_summary_data["average_sl_count"] += sl_count
                    performance_summary_data["potential_profitability"] += (final_deposit - 200)
                    performance_summary_data["total_trades"] += len(predictions)
                    performance_summary_data["successful_trades"] += tp_count
                    performance_summary_data["average_profit_per_trade"] += (final_deposit - 200) / len(predictions) if len(predictions) > 0 else 0

                else:
                    print(f"Échec de l'entraînement pour {symbol} ({temporality})")
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {filename}: {e}")

    total_files = len(performance_time_data)
    if total_files > 0:
        performance_summary_data["average_tp_count"] /= total_files
        performance_summary_data["average_sl_count"] /= total_files
        performance_summary_data["potential_profitability"] /= total_files
        performance_summary_data["average_profit_per_trade"] /= total_files

    if performance_summary_data["total_trades"] > 0:
        performance_summary_data["success_rate"] = performance_summary_data["successful_trades"] / performance_summary_data["total_trades"]

    save_performance_time(PERFORMANCE_TIME_FILE, performance_time_data)
    save_trade_info(TRADE_INFO_FILE, trade_info_data)
    save_performance_summary(PERFORMANCE_SUMMARY_FILE, performance_summary_data)

    print("Traitement terminé.")

def start_flask_server():
    app = Flask(__name__)
    CORS(app)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = False
    app.config['SESSION_KEY_PREFIX'] = ''

    def shutdown_server():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()

    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        shutdown_server()
        return 'Server shutting down...'

    @app.route('/get_config', methods=['GET'])
    def get_config():
        return jsonify(config)

    @app.route('/update_config', methods=['POST'])
    def update_config():
        global config
        data = request.get_json()
        config.update(data)
        return jsonify({"status": "success", "config": config})

    @app.route('/get_volume', methods=['GET'])
    def get_volume():
        return jsonify({"volume": config["volume"]})

    @app.route('/set_volume', methods=['POST'])
    def set_volume():
        global config
        data = request.get_json()
        config["volume"] = float(data['volume'])
        return jsonify({"status": "success"})

    @app.route('/set_block', methods=['GET'])
    def set_block():
        return jsonify({"blocked": config["blocked_all"]})

    @app.route('/unblock', methods=['POST'])
    def unblock():
        global config
        config["blocked_all"] = 0
        return jsonify({"blocked": config["blocked_all"]})

    @app.route('/blockall', methods=['POST'])
    def blockall():
        global config
        config["blocked_all"] = 1
        return jsonify({"blocked": config["blocked_all"]})

    @app.route('/get_block_status', methods=['GET'])
    def get_block_status():
        symbol = request.args.get('symbol')
        status = "blocked" if config["blocked_symbols"].get(symbol, False) else "unblocked"
        return jsonify({"status": status})

    @app.route('/set_block_status', methods=['POST'])
    def set_block_status():
        data = request.get_json()
        symbol = data.get('symbol')
        status = data.get('status')
        if status == "blocked":
            config["blocked_symbols"][symbol] = True
        elif status == "unblocked":
            config["blocked_symbols"][symbol] = False
        return jsonify({"status": "success"})

    def reset_status_after_delay():
        time.sleep(5)
        config["status"] = "success"

    @app.route('/close_all_positions', methods=['GET'])
    def close_all_positions():
        global config
        config["blocked_all"] = 1
        data = request.args
        if data.get('delete') == "True":
            config["status"] = 404
            threading.Thread(target=reset_status_after_delay).start()
            return jsonify({"status": 404})
        else:
            return jsonify({"status": "success"})

        return jsonify({"status": "no action"})

    @app.route('/get_status', methods=['GET'])
    def get_status():
        return jsonify({"status": config["status"]})

    @app.route('/receive_data', methods=['POST'])
    def receive_data():
        global config
        raw_data = request.data.decode('utf-8')
        try:
            data = request.get_json()
            if not data:
                return jsonify({"status": "error", "message": "No data received."}), 400
            if 'datetime' not in data or 'prix' not in data or 'symboles' not in data:
                print({"status": "error", "message": "Missing 'datetime', 'prix' or 'symboles' in request."})
                return jsonify({"status": "error", "message": "Missing 'datetime', 'prix' or 'symboles' in request."}), 400
        except:
            try:
                cleaned_data = str(raw_data).replace('\u0000', '')
                data = json.loads(cleaned_data)
            except Exception as error:
                print({"status": "error", "message": "Invalid JSON format.", "details": str(error)})
                return jsonify({"status": "error", "message": "Invalid JSON format.", "details": str(error)}), 400

        try:
            data['datetime'] = data['datetime'].replace("+", " ")
            current_time = datetime.strptime(data['datetime'], '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            print({"status": "error", "message": f"Date format error: {str(e)}"})
            return jsonify({"status": "error", "message": f"Date format error: {str(e)}"}), 400

        try:
            prix = float(data['prix'])
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid 'prix' value."}), 400

        symbol = data['symboles']
        temporalities = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600
        }

        new_data = None

        for temporality, seconds in temporalities.items():
            filepath = os.path.join(DATA_DIRECT_DIR, f"{symbol}_{temporality}.csv")
            last_candle = get_last_candle(filepath, temporality)
            if last_candle is not None:
                last_time = datetime.strptime(last_candle['datetime'], '%Y-%m-%d %H:%M:%S')
                time_diff = (current_time - last_time).total_seconds()
                if time_diff < seconds:
                    continue

            percentage_change = 0.0
            if last_candle is not None:
                percentage_change = (prix - last_candle['close']) / last_candle['close'] * 100

            new_data = pd.DataFrame([{
                "datetime": current_time,
                "open": prix,
                "high": prix,
                "low": prix,
                "close": prix,
                "volume": config["volume"],
                "time_frame": temporality,
                "day_of_week": current_time.weekday(),
                "hour": current_time.hour,
                "percentage_change": percentage_change,
                "symbol": symbol,
                "temporalite": temporality,
                "sentiment": 0.0  # Mettre 0 par défaut pour les sentiments
            }])
            update_csv_file(filepath, new_data)

        if new_data is not None:
            predictions, tp_count, sl_count, final_deposit = predict_future_candles(new_data, symbol, "M1", config)

            if predictions:
                prediction_result = predictions[0]

                response_data = {
                    'status': config["status"],
                    'volume': config["volume"],
                    'request_id': str(uuid.uuid4()),
                    'predicted_close': prediction_result['entry_price'],
                    'TP': prediction_result['take_profit'],
                    'SL': prediction_result['stop_loss'],
                    'buy_price': prediction_result['entry_price'],
                    'order_type': 1 if prediction_result['direction'] == "buy" else 0,
                    'symboles': symbol
                }

                prediction_filepath = os.path.join(os.getcwd(), "config", f"{symbol}_predictions.json")
                save_trade_info(prediction_filepath, [response_data])

                print(f"Réponse des prédictions envoyée au serveur: {response_data}")

                return jsonify(response_data)
            else:
                return jsonify({"status": "error", "message": "No predictions available."}), 400
        else:
            return jsonify({"status": "error", "message": "No new data to process."}), 400

    def check_m1_files():
        while True:
            time.sleep(60)
            for filename in os.listdir(DATA_DIRECT_DIR):
                if filename.endswith("M1.csv"):
                    filepath = os.path.join(DATA_DIRECT_DIR, filename)
                    data = load_csv_data(filepath)
                    if len(data) >= 50:
                        print("50 lignes atteintes pour M1, mise en pause du serveur pour l'entraînement...")
                        process_files(config)
                        print("Entraînement terminé, redémarrage du serveur...")

    threading.Thread(target=check_m1_files).start()

    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_flask_server)
    server_thread.start()

    def signal_handler(sig, frame):
        print('Shutting down server...')
        requests.post('http://127.0.0.1:5000/shutdown')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server_thread.join()
