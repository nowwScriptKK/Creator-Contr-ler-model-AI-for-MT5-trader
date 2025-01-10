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
import requests
import logging
from ta import add_all_ta_features
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Configurer le logging
logging.basicConfig(level=logging.INFO)

# Télécharger les ressources NLTK
nltk.download('vader_lexicon')

# Répertoires et fichiers de configuration
BASE_DIR = os.path.join(os.getcwd(), "config")
DIRS = {
    "data_direct": os.path.join(BASE_DIR, "dataDirect"),
    "data_trained": os.path.join(BASE_DIR, "dataTrained"),
    "models": os.path.join(BASE_DIR, "models"),
    "symbols_info": os.path.join(BASE_DIR, "symbols_info.json"),
    "performance_time": os.path.join(BASE_DIR, "performance_time.json"),
    "trade_info": os.path.join(BASE_DIR, "trade_info.json"),
    "performance_summary": os.path.join(BASE_DIR, "performance_summary.json")
}

# Configuration
config = {
    "tp_pips": 7,
    "sl_pips": 3.5,
    "volume": 0.1,
    "min_profit_potential": 1.6,
    "model_params": {"lstm_units": 50, "epochs": 100, "batch_size": 32, "patience": 10},
    "status": 1,
    "blocked_all": 0,
    "blocked_symbols": {}
}

# Charger les modèles sauvegardés
def load_saved_models(model_dir):
    paths = {
        "lstm": os.path.join(model_dir, "lstm_model.h5"),
        "scaler": os.path.join(model_dir, "scaler.pkl")
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le modèle {name} n'existe pas à l'emplacement {path}.")
    return load_model(paths["lstm"]), joblib.load(paths["scaler"])

LSTM_MODEL, SCALER = load_saved_models(DIRS["models"])

# Charger les informations des symboles
def load_symbol_info(json_file):
    if not os.path.exists(json_file) or os.path.getsize(json_file) == 0:
        raise FileNotFoundError(f"Le fichier JSON {json_file} n'existe pas ou est vide.")
    with open(json_file, 'r') as f:
        return json.load(f)

# Calculer TP et SL
def calculate_tp_sl(symbol_info, direction, entry_price, tp_pips, sl_pips):
    decimals_per_pip = symbol_info['decimals_per_pip']
    tp_offset = tp_pips / (10 ** decimals_per_pip)
    sl_offset = sl_pips / (10 ** decimals_per_pip)
    if direction == "buy":
        return entry_price + tp_offset, entry_price - sl_offset
    return entry_price - tp_offset, entry_price + sl_offset

# Extraire le symbole et la temporalité du nom de fichier
def extract_symbol_and_temporality(filename):
    match = re.match(r"(.+?)_([M]\d+|H1)(?:_chaine)?\.csv", filename)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(f"Le nom de fichier {filename} ne correspond pas au format attendu.")

# Charger les données CSV
def load_csv_data(filepath):
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"Le fichier {filepath} est vide.")
    return pd.read_csv(filepath)

# Charger les données de trading
def load_trading_data(folder_path):
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and "_" in file:
            parts = file.split("_")

            if "PERIOD" in parts:
                symbol = parts[0]
                temporalite = parts[-1].replace('.csv', '')
            else:
                symbol, temporalite = parts[0], parts[1].replace('.csv', '')

            df = pd.read_csv(os.path.join(folder_path, file))
            if 'datetime' not in df.columns:
                logging.warning(f"Le fichier {file} ne contient pas la colonne 'datetime'. Ignoré.")
                continue

            df['symbol'] = symbol
            df['temporalite'] = temporalite
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y.%m.%d %H:%M')
            dataframes.append(df)

    if not dataframes:
        raise ValueError("Aucun fichier valide trouvé avec une colonne 'datetime'.")

    return pd.concat(dataframes, ignore_index=True)

# Lecture des fichiers textuels et analyse des sentiments
def process_textual_data(folder_path):
    sia = SentimentIntensityAnalyzer()
    text_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and "_" not in file:
            try:
                df = pd.read_csv(os.path.join(folder_path, file), on_bad_lines='skip')
                if {'datetime', 'title', 'info', 'site'}.issubset(df.columns):
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df['sentiment'] = df['info'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
                    text_data.append(df[['datetime', 'sentiment']])
                else:
                    logging.warning(f"Le fichier {file} ne contient pas les colonnes requises. Ignoré.")
            except pd.errors.ParserError as e:
                logging.error(f"Erreur lors de la lecture du fichier {file}: {e}")
                continue
    if not text_data:
        logging.warning("Aucun fichier textuel valide trouvé.")
        return pd.DataFrame(columns=['datetime', 'sentiment'])
    return pd.concat(text_data, ignore_index=True)

# Fusion des sentiments avec les données de trading
def merge_sentiments_with_trading(trading_data, textual_data):
    trading_data.sort_values('datetime', inplace=True)
    textual_data.sort_values('datetime', inplace=True)
    trading_data['sentiment'] = 0
    textual_index = 0
    for idx, row in trading_data.iterrows():
        while textual_index < len(textual_data) - 1 and \
              abs(textual_data.iloc[textual_index + 1]['datetime'] - row['datetime']) < \
              abs(textual_data.iloc[textual_index]['datetime'] - row['datetime']):
            textual_index += 1

        if textual_index < len(textual_data):
            trading_data.at[idx, 'sentiment'] = textual_data.iloc[textual_index]['sentiment']
        else:
            logging.warning(f"Index out of bounds: textual_index={textual_index} for trading_data index={idx}")
    return trading_data

# Préparer les données pour le modèle
def prepare_data_with_sentiments(data, n_steps):
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )

    label_encoder_symbol = LabelEncoder()
    label_encoder_temporalite = LabelEncoder()
    data['symbol'] = label_encoder_symbol.fit_transform(data['symbol'])
    data['temporalite'] = label_encoder_temporalite.fit_transform(data['temporalite'])

    features = data[['open', 'high', 'low', 'close', 'volume', 'sentiment', 'symbol', 'temporalite']].values
    labels = (data['close'].shift(-1) > data['close']).astype(int).values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(len(features_scaled) - n_steps):
        X.append(features_scaled[i:i + n_steps])
        y.append(labels[i + n_steps])
    return np.array(X), np.array(y), scaler

# Construire un modèle LSTM amélioré
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Calculer la direction du marché et ajouter le spread
def calculate_market_direction_with_tp_sl(predictions, data, tp_rate=0.01, sl_rate=0.01):
    directions = []
    for i, pred in enumerate(predictions):
        close_price = data.iloc[i]['close']
        tp = close_price * (1 + tp_rate)
        sl = close_price * (1 - sl_rate)

        if pred > 0.5:
            direction = "buy"
        else:
            direction = "sell"

        spread = np.random.uniform(0.1, 0.5)
        achieved_tp = (data.iloc[i + 1:]['high'] >= tp).any() if direction == "buy" else (data.iloc[i + 1:]['low'] <= tp).any()
        achieved_sl = (data.iloc[i + 1:]['low'] <= sl).any() if direction == "buy" else (data.iloc[i + 1:]['high'] >= sl).any()

        if achieved_tp:
            result = "Take Profit"
        elif achieved_sl:
            result = "Stop Loss"
        else:
            result = "Open"

        directions.append((direction, pred, spread, result))
    return directions

# Déplacer le fichier avant entraînement
def move_file_before_training(filepath, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    try:
        shutil.move(filepath, destination_dir)
        print(f"Fichier {filepath} déplacé vers {destination_dir}")
        return True
    except Exception as e:
        print(f"Erreur lors du déplacement du fichier {filepath} vers {destination_dir}: {e}")
        return False

# Filtrer les trades
def filter_trades(predictions, tp_pips, sl_pips, min_profit_potential):
    return [pred for pred in predictions if abs(pred['take_profit'] - pred['entry_price']) / abs(pred['stop_loss'] - pred['entry_price']) >= min_profit_potential]

# Déterminer la direction du marché
def determine_market_direction(current_price, prediction):
    if prediction > current_price:
        return "buy"
    return "sell"

# Prédire les bougies futures
def predict_future_candles(data, symbol, temporality, config):
    symbol_info = load_symbol_info(DIRS["symbols_info"])
    symbol_data = next((item for item in symbol_info['symbols'] if item['symbol'] == symbol), None)
    if not symbol_data:
        raise ValueError(f"Le symbole {symbol} n'est pas défini dans le fichier JSON.")
    input_data = np.random.rand(len(data), 10, 7)
    predictions_lstm = LSTM_MODEL.predict(input_data)
    tp_count = sl_count = 0
    deposit = 200
    leverage = 1
    predictions = []
    for i, prediction in enumerate(predictions_lstm):
        entry_price = data['close'].iloc[i]
        direction = determine_market_direction(entry_price, prediction[0])
        order_type = 1 if direction == "buy" else 0
        tp, sl = calculate_tp_sl(symbol_data, direction, entry_price, config["tp_pips"], config["sl_pips"])
        if direction == "buy":
            if data['high'].iloc[i+1:].max() >= tp:
                tp_count += 1
                deposit += (tp - entry_price) * leverage
            elif data['low'].iloc[i+1:].min() <= sl:
                sl_count += 1
                deposit -= (entry_price - sl) * leverage
        else:
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
            "order_type": order_type,
            "entry_price": entry_price,
            "take_profit": tp,
            "stop_loss": sl,
            "deposit": deposit
        })
    return filter_trades(predictions, config["tp_pips"], config["sl_pips"], config["min_profit_potential"]), tp_count, sl_count, deposit

# Afficher les bougies
def display_candles(data, temporality):
    print(f"Affichage des bougies pour la temporalité {temporality}:")
    for index, row in data.iterrows():
        print(f"{row['datetime']} - Open: {row['open']}, High: {row['high']}, Low: {row['low']}, Close: {row['close']}, Volume: {row['volume']}")

# Enregistrer les performances de temps
def save_performance_time(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Enregistrer les informations des trades
def save_trade_info(filepath, data):
    existing_data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                existing_data = json.load(f)
                if isinstance(existing_data, dict):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    existing_data.extend(data)
    with open(filepath, 'w') as f:
        json.dump(existing_data, f, indent=4)

# Enregistrer les moyennes des performances
def save_performance_summary(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Obtenir la dernière bougie de la bonne temporalité
def get_last_candle(filepath, temporality):
    if not os.path.exists(filepath):
        return None
    data = load_csv_data(filepath)
    if data.empty:
        return None
    return data.iloc[-1]

# Créer ou mettre à jour les fichiers CSV
def update_csv_file(filepath, data):
    if not os.path.exists(filepath):
        data.to_csv(filepath, index=False)
    else:
        try:
            existing_data = pd.read_csv(filepath)
            combined_data = pd.concat([existing_data, data]).drop_duplicates().reset_index(drop=True)
            combined_data.to_csv(filepath, index=False)
        except pd.errors.EmptyDataError:
            data.to_csv(filepath, index=False)

# Pipeline complet amélioré
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
    processed_files = set()
    symbols_to_train = set()
    n_steps = 10

    while True:
        files_to_process = [f for f in os.listdir(DIRS["data_direct"]) if f.endswith(".csv") and f not in processed_files]
        if not files_to_process:
            time.sleep(60)
            continue

        try:
            trading_data = load_trading_data(DIRS["data_direct"])
            textual_data = process_textual_data(DIRS["data_direct"])
            if textual_data.empty:
                logging.warning("Aucune donnée textuelle valide trouvée. Utilisation de valeurs par défaut.")
                textual_data = pd.DataFrame(columns=['datetime', 'sentiment'])

            merged_data = merge_sentiments_with_trading(trading_data, textual_data)
            X, y, scaler = prepare_data_with_sentiments(merged_data, n_steps)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            lstm_model = build_lstm(X_train.shape[1:])
            lstm_model.fit(X_train, y_train, epochs=config["model_params"]["epochs"], batch_size=config["model_params"]["batch_size"], validation_split=0.2)
            predictions = lstm_model.predict(X_test).flatten()
            market_directions = calculate_market_direction_with_tp_sl(predictions, merged_data.iloc[-len(X_test):])

            for direction, prob, spread, result in market_directions[:10]:
                logging.info(f"Direction: {direction}, Probabilité: {prob:.2f}, Spread: {spread:.2f}, Résultat: {result}")

            processed_files.update(files_to_process)

        except Exception as e:
            logging.error(f"Erreur lors du traitement des fichiers : {e}")
            time.sleep(60)

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
        config["blocked_symbols"][symbol] = status == "blocked"
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
        return jsonify({"status": "success"})

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
            filepath = os.path.join(DIRS["data_direct"], f"{symbol}_{temporality}.csv")
            try:
                last_candle = get_last_candle(filepath, temporality)
            except ValueError as e:
                print(f"Error: {e}")
                continue

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
                "percentage_change": percentage_change
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
            for filename in os.listdir(DIRS["data_direct"]):
                if filename.endswith("M1.csv"):
                    filepath = os.path.join(DIRS["data_direct"], filename)
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
