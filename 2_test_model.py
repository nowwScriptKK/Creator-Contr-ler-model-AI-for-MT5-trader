import os, re, pandas as pd, numpy as np, shutil, joblib, json, time, threading, uuid
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import signal, sys

# Répertoires et configuration
BASE_DIR = os.path.join(os.getcwd(), "config")
DIRS = {
    "DATA_DIRECT": os.path.join(BASE_DIR, "dataDirect"),
    "DATA_TRAINED": os.path.join(BASE_DIR, "dataTrained"),
    "MODEL": os.path.join(BASE_DIR, "models"),
    "SYMBOL_INFO": os.path.join(BASE_DIR, "symbols_info.json"),
    "PERFORMANCE_TIME": os.path.join(BASE_DIR, "performance_time.json"),
    "TRADE_INFO": os.path.join(BASE_DIR, "trade_info.json")
}

CONFIG = {
    "tp_pips": 8, "sl_pips": 3.6, "volume": 1, "min_profit_potential": 1.2,
    "model_params": {"lstm_units": 50, "epochs": 100, "batch_size": 32, "patience": 10},
    "status": 1, "blocked_all": 0, "blocked_symbols": {}
}

def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"File {filepath} is empty.")
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_saved_models(model_dir):
    paths = ["lstm_model.h5", "xgb_model.pkl", "scaler.pkl"]
    models = []
    for path in paths:
        full_path = os.path.join(model_dir, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model {full_path} does not exist.")
        models.append(load_model(full_path) if path.endswith(".h5") else joblib.load(full_path))
    return models

LSTM_MODEL, XGB_MODEL, SCALER = load_saved_models(DIRS["MODEL"])

def calculate_tp_sl(symbol_info, direction, entry_price, tp_pips, sl_pips):
    decimals_per_pip = symbol_info['decimals_per_pip']
    tp_offset = tp_pips / (10 ** decimals_per_pip)
    sl_offset = sl_pips / (10 ** decimals_per_pip)
    tp = entry_price + tp_offset if direction == "buy" else entry_price - tp_offset
    sl = entry_price - sl_offset if direction == "buy" else entry_price + sl_offset
    return tp, sl

def extract_symbol_and_temporality(filename):
    match = re.match(r"(.+?)_([M]\d+|H1)(?:_chaine)?\.csv", filename)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(f"Filename {filename} does not match expected format.")

def load_csv_data(filepath):
    return pd.read_csv(filepath)

def train_model(data, symbol, temporality, config):
    print(f"Training model for {symbol} with temporality {temporality}")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume', 'percentage_change']])
    X, y = [], []
    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i])
        y.append(scaled_data[i, 3])
    X, y = np.array(X), np.array(y)
    early_stopping = EarlyStopping(monitor='val_loss', patience=config["model_params"]["patience"], restore_best_weights=True)
    LSTM_MODEL.fit(X, y, epochs=config["model_params"]["epochs"], batch_size=config["model_params"]["batch_size"], validation_split=0.2, callbacks=[early_stopping])
    LSTM_MODEL.save(os.path.join(DIRS["MODEL"], "lstm_model.h5"))
    joblib.dump(scaler, os.path.join(DIRS["MODEL"], "scaler.pkl"))
    print(f"Training completed for {symbol} with temporality {temporality}")
    return True

def move_file_before_training(filepath, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    try:
        shutil.move(filepath, destination_dir)
        print(f"File {filepath} moved to {destination_dir}")
        return True
    except Exception as e:
        print(f"Error moving file {filepath} to {destination_dir}: {e}")
        return False

def filter_trades(predictions, tp_pips, sl_pips, min_profit_potential):
    return [pred for pred in predictions if calculate_tp_sl(pred['symbol_info'], pred['direction'], pred['entry_price'], tp_pips, sl_pips)[0] / calculate_tp_sl(pred['symbol_info'], pred['direction'], pred['entry_price'], tp_pips, sl_pips)[1] >= min_profit_potential]

def predict_future_candles(data, symbol, temporality, config):
    symbol_info = load_json(DIRS["SYMBOL_INFO"])
    symbol_data = next((item for item in symbol_info['symbols'] if item['symbol'] == symbol), None)
    if not symbol_data:
        raise ValueError(f"Symbol {symbol} not defined in JSON file.")
    input_data = np.random.rand(len(data), 10, 7)
    predictions_lstm = LSTM_MODEL.predict(input_data)
    tp_count = sl_count = 0
    deposit = 200
    leverage = 1
    predictions = []
    for i, prediction in enumerate(predictions_lstm):
        entry_price = data['close'].iloc[i]
        direction = "buy" if prediction[0] > 0.5 else "sell"
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
            "symbol": symbol, "symbol_info": symbol_data, "direction": direction, "order_type": order_type,
            "entry_price": entry_price, "take_profit": tp, "stop_loss": sl, "deposit": deposit
        })
    filtered_predictions = filter_trades(predictions, config["tp_pips"], config["sl_pips"], config["min_profit_potential"])
    return filtered_predictions, tp_count, sl_count, deposit

def display_candles(data, temporality):
    print(f"Displaying candles for temporality {temporality}:")
    for index, row in data.iterrows():
        print(f"{row['datetime']} - Open: {row['open']}, High: {row['high']}, Low: {row['low']}, Close: {row['close']}, Volume: {row['volume']}")

def get_last_candle(filepath, temporality):
    if not os.path.exists(filepath):
        return None
    data = load_csv_data(filepath)
    if data.empty:
        return None
    return data.iloc[-1]

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

def process_files(config):
    performance_time_data = []
    trade_info_data = []
    processed_files = set()
    while True:
        files_to_process = [f for f in os.listdir(DIRS["DATA_DIRECT"]) if f.endswith(".csv") and f not in processed_files]
        if not files_to_process:
            print("No files to process. Waiting for new files...")
            time.sleep(60)
            continue
        for filename in files_to_process:
            filepath = os.path.join(DIRS["DATA_DIRECT"], filename)
            try:
                print(f"Processing file {filename}...")
                symbol, temporality = extract_symbol_and_temporality(filename)
                #print(f"Symbol: {symbol}, Temporality: {temporality}")
                data = load_csv_data(filepath)
                #print(f"Data loaded: {len(data)} rows")
                if (temporality == "M1" and len(data) < 50) or (temporality != "M1" and len(data) < 5):
                    #print(f"File {filename} has fewer rows than required. Ignored.")
                    continue
                display_candles(data, temporality)
                if move_file_before_training(filepath, DIRS["DATA_TRAINED"]):
                    new_filepath = os.path.join(DIRS["DATA_TRAINED"], filename)
                    start_time = time.time()
                    if train_model(data, symbol, temporality, config):
                        predictions, tp_count, sl_count, final_deposit = predict_future_candles(data, symbol, temporality, config)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        performance_time_data.append({
                            "symbol": symbol, "temporality": temporality, "elapsed_time": elapsed_time
                        })
                        trade_info_data.extend(predictions)
                        print(f"Predictions for {symbol} ({temporality}):")
                        for pred in predictions:
                            print(pred)
                        print(f"Number of TP reached: {tp_count}, Number of SL reached: {sl_count}")
                        print(f"Final deposit: {final_deposit}€")
                        processed_files.add(filename)
                    else:
                        print(f"Training failed for {symbol} ({temporality})")
                else:
                    print(f"Error: File {filename} was not moved correctly.")
                    continue
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
        save_json(DIRS["PERFORMANCE_TIME"], performance_time_data)
        save_json(DIRS["TRADE_INFO"], trade_info_data)
        print("Processing completed.")

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
        return jsonify(CONFIG)

    @app.route('/update_config', methods=['POST'])
    def update_config():
        global CONFIG
        CONFIG.update(request.get_json())
        return jsonify({"status": "success", "config": CONFIG})

    @app.route('/get_volume', methods=['GET'])
    def get_volume():
        return jsonify({"volume": CONFIG["volume"]})

    @app.route('/set_volume', methods=['POST'])
    def set_volume():
        global CONFIG
        CONFIG["volume"] = float(request.get_json()['volume'])
        return jsonify({"status": "success"})

    @app.route('/set_block', methods=['GET'])
    def set_block():
        return jsonify({"blocked": CONFIG["blocked_all"]})

    @app.route('/unblock', methods=['POST'])
    def unblock():
        global CONFIG
        CONFIG["blocked_all"] = 0
        return jsonify({"blocked": CONFIG["blocked_all"]})

    @app.route('/blockall', methods=['POST'])
    def blockall():
        global CONFIG
        CONFIG["blocked_all"] = 1
        return jsonify({"blocked": CONFIG["blocked_all"]})

    @app.route('/get_block_status', methods=['GET'])
    def get_block_status():
        symbol = request.args.get('symbol')
        status = "blocked" if CONFIG["blocked_symbols"].get(symbol, False) else "unblocked"
        return jsonify({"status": status})

    @app.route('/set_block_status', methods=['POST'])
    def set_block_status():
        data = request.get_json()
        symbol = data.get('symbol')
        status = data.get('status')
        CONFIG["blocked_symbols"][symbol] = status == "blocked"
        return jsonify({"status": "success"})

    def reset_status_after_delay():
        time.sleep(5)
        CONFIG["status"] = "success"

    @app.route('/close_all_positions', methods=['GET'])
    def close_all_positions():
        global CONFIG
        CONFIG["blocked_all"] = 1
        if request.args.get('delete') == "True":
            CONFIG["status"] = 404
            threading.Thread(target=reset_status_after_delay).start()
            return jsonify({"status": 404})
        return jsonify({"status": "success"})

    @app.route('/get_status', methods=['GET'])
    def get_status():
        return jsonify({"status": CONFIG["status"]})

    @app.route('/receive_data', methods=['POST'])
    def receive_data():
        global CONFIG
        raw_data = request.data.decode('utf-8')
        try:
            data = request.get_json()
            if not data:
                return jsonify({"status": "error", "message": "No data received."}), 400
            if 'datetime' not in data or 'prix' not in data or 'symboles' not in data:
                return jsonify({"status": "error", "message": "Missing 'datetime', 'prix' or 'symboles' in request."}), 400
        except:
            try:
                cleaned_data = str(raw_data).replace('\u0000', '')
                data = json.loads(cleaned_data)
            except Exception as error:
                return jsonify({"status": "error", "message": "Invalid JSON format.", "details": str(error)}), 400

        try:
            data['datetime'] = data['datetime'].replace("+", " ")
            current_time = datetime.strptime(data['datetime'], '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            return jsonify({"status": "error", "message": f"Date format error: {str(e)}"}), 400

        try:
            prix = float(data['prix'])
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid 'prix' value."}), 400

        symbol = data['symboles']
        temporalities = {
            "M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600
        }

        new_data = None

        for temporality, seconds in temporalities.items():
            filepath = os.path.join(DIRS["DATA_DIRECT"], f"{symbol}_{temporality}.csv")
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
                "volume": CONFIG["volume"],
                "time_frame": temporality,
                "day_of_week": current_time.weekday(),
                "hour": current_time.hour,
                "percentage_change": percentage_change
            }])
            update_csv_file(filepath, new_data)

        if new_data is not None:
            predictions, tp_count, sl_count, final_deposit = predict_future_candles(new_data, symbol, "M1", CONFIG)

            if predictions:
                prediction_result = predictions[0]

                response_data = {
                    'status': CONFIG["status"],
                    'volume': CONFIG["volume"],
                    'request_id': str(uuid.uuid4()),
                    'predicted_close': prediction_result['entry_price'],
                    'TP': prediction_result['take_profit'],
                    'SL': prediction_result['stop_loss'],
                    'buy_price': prediction_result['entry_price'],
                    'order_type': 1 if prediction_result['direction'] == "buy" else 0,
                    'symboles': symbol
                }

                prediction_filepath = os.path.join(os.getcwd(), "config", f"{symbol}_predictions.json")
                save_json(prediction_filepath, [response_data])

                print(f"Prediction response sent to server: {response_data}")

                return jsonify(response_data)
            else:
                return jsonify({"status": "error", "message": "No predictions available."}), 400
        else:
            return jsonify({"status": "error", "message": "No new data to process."}), 400

    def check_m1_files():
        while True:
            time.sleep(60)
            for filename in os.listdir(DIRS["DATA_DIRECT"]):
                if filename.endswith("M1.csv"):
                    filepath = os.path.join(DIRS["DATA_DIRECT"], filename)
                    data = load_csv_data(filepath)
                    if len(data) >= 50:
                        print("50 rows reached for M1, pausing server for training...")
                        process_files(CONFIG)
                        print("Training completed, restarting server...")

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
