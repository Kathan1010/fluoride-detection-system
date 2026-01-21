import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import load_model

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_excel("fluoride_2023.xlsx", engine="openpyxl")

df = df[['pH', 'EC (µS/cm at', 'Total Hardness', 'F (mg/L)']]
df.columns = ['pH', 'EC', 'Hardness', 'Fluoride']

df.replace('-', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

print("Dataset size:", df.shape)

# -------------------------------
# 2. FEATURE - TARGET SPLIT
# -------------------------------
X = df[['pH', 'EC', 'Hardness']]
y = df[['Fluoride']]

# -------------------------------
# 3. LOAD SCALERS
# -------------------------------
scaler_X = pickle.load(open("scaler_X.pkl", "rb"))
scaler_y = pickle.load(open("scaler_y.pkl", "rb"))

X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# -------------------------------
# 4. CREATE LSTM SEQUENCES
# -------------------------------
def create_sequences(X, y, window=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

X_lstm, y_lstm = create_sequences(X_scaled, y_scaled)

print("LSTM input shape:", X_lstm.shape)

# -------------------------------
# 5. LOAD TRAINED MODEL
# -------------------------------
model = load_model("fluoride_model.h5", compile=False)
print("Model loaded successfully.")

# -------------------------------
# 6. EVALUATE MODEL
# -------------------------------
y_pred_scaled = model.predict(X_lstm)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_lstm)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("\nMODEL PERFORMANCE")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# -------------------------------
# 7. PREDICT NEW SAMPLE
# -------------------------------
new_sample = {
    "pH": 7.96,
    "EC": 395,
    "Hardness": 160
}

X_new = np.array([[new_sample["pH"],
                   new_sample["EC"],
                   new_sample["Hardness"]]])

X_new_scaled = scaler_X.transform(X_new)

X_seq = np.repeat(X_new_scaled.reshape(1,1,3), 10, axis=1)

pred_scaled = model.predict(X_seq)
pred_fluoride = scaler_y.inverse_transform(pred_scaled)

print("\nPredicted Fluoride (mg/L):", pred_fluoride[0][0])
