import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import joblib
from ta import add_all_ta_features
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import logging
from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd

# Configurer le logging
logging.basicConfig(level=logging.INFO)

# Télécharger les ressources NLTK
nltk.download('vader_lexicon')

# Configurer TensorFlow pour utiliser toutes les ressources disponibles
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Activer le calcul en précision mixte pour améliorer les performances
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Étape 1 : Charger les données de trading
def load_trading_data(folder_path):
    def process_file(file):
        parts = file.split("_")
        if "PERIOD" in parts:
            symbol = parts[0]
            temporalite = parts[-1].replace('.csv', '')
        else:
            symbol, temporalite = parts[0], parts[1].replace('.csv', '')

        df = pd.read_csv(os.path.join(folder_path, file))
        if 'datetime' not in df.columns:
            logging.warning(f"Le fichier {file} ne contient pas la colonne 'datetime'. Ignoré.")
            return None

        df['symbol'] = symbol
        df['temporalite'] = temporalite
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y.%m.%d %H:%M')
        logging.info(f"Fichier {file} chargé avec succès.")
        return df

    with ThreadPoolExecutor() as executor:
        files = [file for file in os.listdir(folder_path) if file.endswith(".csv") and "_" in file]
        dataframes = list(executor.map(process_file, files))

    dataframes = [df for df in dataframes if df is not None]
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
                    logging.info(f"Fichier {file} traité avec succès.")
                else:
                    logging.warning(f"Le fichier {file} ne contient pas les colonnes requises. Ignoré.")
            except pd.errors.ParserError as e:
                logging.error(f"Erreur lors de la lecture du fichier {file}: {e}")
                continue
    if not text_data:
        logging.warning("Aucun fichier textuel valide trouvé.")
        return pd.DataFrame(columns=['datetime', 'sentiment'])
    return pd.concat(text_data, ignore_index=True)

# Étape 3 : Associer les sentiments aux données de trading
def merge_sentiments_with_trading(trading_data, textual_data):
    if textual_data.empty:
        logging.info("Aucune donnée textuelle trouvée. Continuation sans sentiments.")
        return trading_data

    trading_data.sort_values('datetime', inplace=True)
    textual_data.sort_values('datetime', inplace=True)
    trading_data['sentiment'] = 0.0
    textual_index = 0
    for idx, row in trading_data.iterrows():
        while textual_index < len(textual_data) - 1 and \
              textual_data.iloc[textual_index + 1]['datetime'] < row['datetime']:
            textual_index += 1
        trading_data.at[idx, 'sentiment'] = float(textual_data.iloc[textual_index]['sentiment'])
    logging.info("Sentiments associés aux données de trading avec succès.")
    return trading_data

# Étape 4 : Préparer les données pour le modèle
def prepare_data_with_sentiments(data, n_steps, has_sentiments):
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )

    label_encoder_symbol = LabelEncoder()
    label_encoder_temporalite = LabelEncoder()
    data['symbol'] = label_encoder_symbol.fit_transform(data['symbol'])
    data['temporalite'] = label_encoder_temporalite.fit_transform(data['temporalite'])

    if has_sentiments:
        data['sentiment'].fillna(0, inplace=True)
        features = data[['open', 'high', 'low', 'close', 'volume', 'sentiment', 'symbol', 'temporalite']].values
    else:
        features = data[['open', 'high', 'low', 'close', 'volume', 'symbol', 'temporalite']].values

    labels = (data['close'].shift(-1) > data['close']).astype(int).values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(len(features_scaled) - n_steps):
        X.append(features_scaled[i:i + n_steps])
        y.append(labels[i + n_steps])
    logging.info("Données préparées pour le modèle avec succès.")
    return np.array(X), np.array(y), scaler

# Étape 5 : Construire un modèle LSTM
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logging.info("Modèle LSTM construit avec succès.")
    return model

# Étape 6 : Entraîner un modèle XGBoost
def train_xgboost(X, y):
    model = xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=100, max_depth=5, learning_rate=0.1
    )
    model.fit(X, y)
    logging.info("Modèle XGBoost entraîné avec succès.")
    return model

# Étape 7 : Sauvegarder les modèles et le scaler
def save_models(lstm_model, xgb_model, scaler, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    lstm_model.save(os.path.join(output_dir, "lstm_model.h5"))
    joblib.dump(xgb_model, os.path.join(output_dir, "xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    logging.info("Modèles sauvegardés avec succès.")

# Étape 8 : Charger les modèles sauvegardés
def load_saved_models(model_dir="models"):
    lstm_model = load_model(os.path.join(model_dir, "lstm_model.h5"))
    xgb_model = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    logging.info("Modèles chargés avec succès.")
    return lstm_model, xgb_model, scaler

# Étape 9 : Pipeline complet
def main():
    folder_path = "C:\\Users\\maxwi\\Desktop\\ModelAI_V2.0\\models\\entrainement"
    logging.info("Chargement des données de trading...")
    trading_data = load_trading_data(folder_path)
    logging.info("Chargement des données textuelles...")
    textual_data = process_textual_data(folder_path)
    has_sentiments = not textual_data.empty
    logging.info("Association des sentiments aux données de trading...")
    merged_data = merge_sentiments_with_trading(trading_data, textual_data)
    n_steps = 10
    logging.info("Préparation des données pour le modèle...")
    X, y, scaler = prepare_data_with_sentiments(merged_data, n_steps, has_sentiments)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Construction du modèle LSTM...")
    lstm_model = build_lstm(X_train.shape[1:])
    logging.info("Entraînement du modèle LSTM...")
    lstm_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)
    logging.info("Prédiction avec le modèle LSTM...")
    lstm_preds = lstm_model.predict(X_test).flatten()
    lstm_preds = (lstm_preds > 0.5).astype(int)
    logging.info("Entraînement du modèle XGBoost...")
    xgb_model = train_xgboost(lstm_preds.reshape(-1, 1), y_test)
    logging.info("Prédiction avec le modèle XGBoost...")
    xgb_preds = xgb_model.predict(lstm_preds.reshape(-1, 1))
    logging.info(f"Accuracy: {accuracy_score(y_test, xgb_preds)}")
    logging.info("Sauvegarde des modèles...")
    save_models(lstm_model, xgb_model, scaler)
    logging.info("Fin du programme.")

if __name__ == "__main__":
    main()
