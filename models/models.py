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

# Configurer le logging
logging.basicConfig(level=logging.INFO)

# Télécharger les ressources NLTK
nltk.download('vader_lexicon')

# Étape 1 : Charger les données de trading
def load_trading_data(folder_path):
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and "_" in file:
            parts = file.split("_")

            # Gestion des cas avec ou sans "PERIOD"
            if "PERIOD" in parts:
                symbol = parts[0]
                temporalite = parts[-1].replace('.csv', '')  # Récupère la dernière partie comme temporalité
            else:
                symbol, temporalite = parts[0], parts[1].replace('.csv', '')

            df = pd.read_csv(os.path.join(folder_path, file))
            if 'datetime' not in df.columns:
                logging.warning(f"Le fichier {file} ne contient pas la colonne 'datetime'. Ignoré.")
                continue  # Passer au fichier suivant

            # Convertir 'datetime' en datetime pandas avec le format correct
            df['symbol'] = symbol
            df['temporalite'] = temporalite
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y.%m.%d %H:%M')  # Format corrigé
            dataframes.append(df)

    if not dataframes:
        raise ValueError("Aucun fichier valide trouvé avec une colonne 'datetime'.")

    return pd.concat(dataframes, ignore_index=True)

# Étape 2 : Lecture des fichiers textuels et analyse des sentiments
def process_textual_data(folder_path):
    sia = SentimentIntensityAnalyzer()
    text_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and "_" not in file:  # Fichiers non liés aux symboles/temporalité
            try:
                df = pd.read_csv(os.path.join(folder_path, file), on_bad_lines='skip')
                if {'datetime', 'title', 'info', 'site'}.issubset(df.columns):
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    # Calcul des sentiments
                    df['sentiment'] = df['info'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
                    text_data.append(df[['datetime', 'sentiment']])
                else:
                    logging.warning(f"Le fichier {file} ne contient pas les colonnes requises. Ignoré.")
            except pd.errors.ParserError as e:
                logging.error(f"Erreur lors de la lecture du fichier {file}: {e}")
                continue
    if not text_data:
        logging.warning("Aucun fichier textuel valide trouvé.")
        return pd.DataFrame(columns=['datetime', 'sentiment'])  # Retourner un DataFrame vide si aucun fichier valide n'est trouvé
    return pd.concat(text_data, ignore_index=True)

# Étape 3 : Associer les sentiments aux données de trading
def merge_sentiments_with_trading(trading_data, textual_data):
    trading_data.sort_values('datetime', inplace=True)
    textual_data.sort_values('datetime', inplace=True)
    trading_data['sentiment'] = 0  # Valeur par défaut
    textual_index = 0
    for idx, row in trading_data.iterrows():
        while textual_index < len(textual_data) - 1 and \
              abs(textual_data.iloc[textual_index + 1]['datetime'] - row['datetime']) < \
              abs(textual_data.iloc[textual_index]['datetime'] - row['datetime']):
            textual_index += 1
        trading_data.at[idx, 'sentiment'] = textual_data.iloc[textual_index]['sentiment']
    return trading_data

# Étape 4 : Préparer les données pour le modèle
def prepare_data_with_sentiments(data, n_steps):
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )

    # Encoder les colonnes 'symbol' et 'temporalite'
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
    return model

# Étape 6 : Entraîner un modèle XGBoost
def train_xgboost(X, y):
    model = xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=100, max_depth=5, learning_rate=0.1
    )
    model.fit(X, y)
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
    return lstm_model, xgb_model, scaler

# Étape 9 : Pipeline complet
def main():
    folder_path = "C:\\Users\\maxwi\\Desktop\\ModelAI_V2.0\\models\\entrainement"
    trading_data = load_trading_data(folder_path)
    textual_data = process_textual_data(folder_path)
    if textual_data.empty:
        logging.error("Aucune donnée textuelle valide trouvée. Arrêt du programme.")
        return
    merged_data = merge_sentiments_with_trading(trading_data, textual_data)
    n_steps = 10
    X, y, scaler = prepare_data_with_sentiments(merged_data, n_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lstm_model = build_lstm(X_train.shape[1:])
    lstm_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)
    lstm_preds = lstm_model.predict(X_test).flatten()
    lstm_preds = (lstm_preds > 0.5).astype(int)
    xgb_model = train_xgboost(lstm_preds.reshape(-1, 1), y_test)
    xgb_preds = xgb_model.predict(lstm_preds.reshape(-1, 1))
    logging.info(f"Accuracy: {accuracy_score(y_test, xgb_preds)}")
    save_models(lstm_model, xgb_model, scaler)

if __name__ == "__main__":
    main()
