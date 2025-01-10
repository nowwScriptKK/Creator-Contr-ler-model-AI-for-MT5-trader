import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
import xgboost as xgb
import joblib
from ta import add_all_ta_features
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO)

# Télécharger les ressources NLTK
nltk.download('vader_lexicon')

# Charger les informations des symboles
def load_symbols_info(file_path):
    with open(file_path, 'r') as file:
        symbols_info = json.load(file)
    return {symbol['symbol']: symbol for symbol in symbols_info['symbols']}

# Étape 1 : Charger les données de trading
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

# Étape 2 : Lecture des fichiers textuels et analyse des sentiments
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

def merge_sentiments_with_trading(trading_data, textual_data):
    trading_data.sort_values('datetime', inplace=True)
    textual_data.sort_values('datetime', inplace=True)
    trading_data['sentiment'] = 0
    textual_index = 0
    for idx, row in trading_data.iterrows():
        # Assurez-vous que textual_index reste dans les limites de textual_data
        while textual_index < len(textual_data) - 1 and \
              abs(textual_data.iloc[textual_index + 1]['datetime'] - row['datetime']) < \
              abs(textual_data.iloc[textual_index]['datetime'] - row['datetime']):
            textual_index += 1

        if textual_index < len(textual_data):  # Vérifiez si l'indice est valide
            trading_data.at[idx, 'sentiment'] = textual_data.iloc[textual_index]['sentiment']
        else:
            logging.warning(f"Index out of bounds: textual_index={textual_index} for trading_data index={idx}")
    return trading_data

# Étape 4 : Préparer les données pour le modèle
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

# Étape 5 : Construire un modèle LSTM amélioré
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

# Étape 6 : Calculer la direction du marché et ajouter le spread
def calculate_market_direction_with_tp_sl(predictions, data, symbols_info, tp_rate=0.01, sl_rate=0.0033):
    directions = []
    for i, pred in enumerate(predictions):
        symbol = data.iloc[i]['symbol']
        close_price = data.iloc[i]['close']
        symbol_info = symbols_info.get(symbol)
        if symbol_info:
            decimals_per_pip = symbol_info['decimals_per_pip']
            tp_pips = np.random.uniform(1, 100)
            sl_pips = tp_pips / 3
            tp = close_price + (tp_pips * 10 ** (-decimals_per_pip))
            sl = close_price - (sl_pips * 10 ** (-decimals_per_pip))
        else:
            tp = close_price * (1 + tp_rate)  # Niveau de Take Profit
            sl = close_price * (1 - sl_rate)  # Niveau de Stop Loss

        if pred > 0.5:  # Achat si probabilité > 0.5
            direction = "buy"
        else:  # Vente sinon
            direction = "sell"

        spread = np.random.uniform(0.1, 0.5)  # Spread simulé
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

# Étape 7 : Entraîner un modèle XGBoost
def train_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model

# Étape 8 : Tests en conditions réelles
def simulate_real_conditions(predictions, data, symbols_info, initial_balance=1000, leverage=30, lot_size=0.01):
    balance = initial_balance
    equity = initial_balance
    positions = []

    for i, (direction, prob, spread, result) in enumerate(predictions):
        symbol = data.iloc[i]['symbol']
        close_price = data.iloc[i]['close']
        symbol_info = symbols_info.get(symbol)
        if symbol_info:
            decimals_per_pip = symbol_info['decimals_per_pip']
            tp_pips = np.random.uniform(1, 100)
            sl_pips = tp_pips / 3
            tp_price_diff = tp_pips * 10 ** (-decimals_per_pip)
            sl_price_diff = sl_pips * 10 ** (-decimals_per_pip)
        else:
            tp_price_diff = close_price * 0.01
            sl_price_diff = close_price * 0.0033

        position_size = equity * leverage * lot_size

        if direction == "buy":
            entry_price = close_price * (1 + spread / 100)
            tp_price = entry_price + tp_price_diff
            sl_price = entry_price - sl_price_diff
        else:
            entry_price = close_price * (1 - spread / 100)
            tp_price = entry_price - tp_price_diff
            sl_price = entry_price + sl_price_diff

        positions.append({
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'position_size': position_size,
            'result': result
        })

        if result == "Take Profit":
            equity += position_size * (tp_price - entry_price) / entry_price
        elif result == "Stop Loss":
            equity -= position_size * (entry_price - sl_price) / entry_price

    return equity, positions

# Fonction pour enregistrer les données de trade
def save_trade_data(epoch, predictions, data, symbols_info, file_path):
    market_directions = calculate_market_direction_with_tp_sl(predictions, data, symbols_info)
    trade_data = []
    for i, (direction, prob, spread, result) in enumerate(market_directions):
        trade_data.append({
            'epoch': epoch,
            'symbol': data.iloc[i]['symbol'],
            'datetime': data.iloc[i]['datetime'],
            'direction': direction,
            'probability': prob,
            'spread': spread,
            'result': result
        })
    df = pd.DataFrame(trade_data)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

# Étape 9 : Pipeline complet amélioré
def main():
    folder_path = "C:\\Users\\maxwi\\Desktop\\ModelAI_V2.0\\models\\entrainement"
    symbols_info_path = os.path.join(folder_path, 'symbols_info.json')
    symbols_info = load_symbols_info(symbols_info_path)

    trading_data = load_trading_data(folder_path)
    textual_data = process_textual_data(folder_path)
    if textual_data.empty:
        logging.warning("Aucune donnée textuelle valide trouvée. Le modèle sera entraîné uniquement avec les données boursières.")
        textual_data = pd.DataFrame(columns=['datetime', 'sentiment'])
    merged_data = merge_sentiments_with_trading(trading_data, textual_data)
    n_steps = 10
    X, y, scaler = prepare_data_with_sentiments(merged_data, n_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lstm_model = build_lstm(X_train.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Callback pour enregistrer les données de trade à chaque époque
    trade_data_file = 'trade_data.csv'
    save_trade_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: save_trade_data(epoch, lstm_model.predict(X_train).flatten(), merged_data.iloc[:len(X_train)], symbols_info, trade_data_file))

    # Ajout de logs pour vérifier les époques
    class LogEpochs(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logging.info(f"Fin de l'époque {epoch + 1}, perte: {logs['loss']}, précision: {logs['accuracy']}, perte de validation: {logs['val_loss']}, précision de validation: {logs['val_accuracy']}")

    lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping, save_trade_callback, LogEpochs()])

    lstm_model.save('lstm_model.h5')
    joblib.dump(scaler, 'scaler.pkl')

    predictions = lstm_model.predict(X_test).flatten()
    market_directions = calculate_market_direction_with_tp_sl(predictions, merged_data.iloc[-len(X_test):], symbols_info)

    xgb_model = train_xgboost(X_train.reshape(X_train.shape[0], -1), y_train)
    joblib.dump(xgb_model, 'xgb_model.pkl')

    final_equity, positions = simulate_real_conditions(market_directions, merged_data.iloc[-len(X_test):], symbols_info)
    logging.info(f"Équité finale: {final_equity}")

    for position in positions[:10]:
        logging.info(f"Position: {position}")

    # Afficher les symboles testés avec le TP ou le SL atteint et le gain/perte pour chaque trade
    for symbol in merged_data['symbol'].unique():
        symbol_positions = [pos for pos in positions if pos['symbol'] == symbol]
        logging.info(f"Symbol: {symbol}")
        for pos in symbol_positions[:10]:  # Afficher les 10 premiers trades pour chaque symbole
            logging.info(f"  {pos}")

if __name__ == "__main__":
    main()
