# ================================
# Imports
# ================================
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.layers import BatchNormalization, GaussianNoise, Add, Input, Dropout
from tensorflow.keras.models import Model
import keras_tuner as kt
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

# Register Mish activation
def mish(x):
    return x * K.tanh(K.softplus(x))

get_custom_objects().update({'mish': mish})

# ================================
# Utility Functions
# ================================
def remove_outliers_iqr(df, features):
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

def preprocess(data, test_dataset=False):
    data.columns = [re.sub(r"\s*\([^)]*\)", "", col).strip() for col in data.columns]
    data = data.drop(["Unnamed: 0"], axis="columns", errors="ignore")
    data = data.dropna()

    data = data[
        (data["Tank Width"] > 0)
        & (data["Tank Length"] > 0)
        & (data["Tank Height"] > 0)
        & (data["Vapour Height"] >= 0)
        & (data["Vapour Temperature"] > 0)
        & (data["Liquid Temperature"] > 0)
    ]

    if not test_dataset:
        data = data.drop_duplicates()

    data["Status"] = data["Status"].str.lower().str.replace(" ", "")
    data["Status"] = data["Status"].apply(
        lambda x: "superheated" if "h" in x else ("subcooled" if "c" in x else x)
    )

    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    if not test_dataset:
        data = remove_outliers_iqr(data, numeric_cols)

    data["Liquid Boiling Temperature"] += 273.15
    data["Liquid Critical Temperature"] += 273.15
    data["Tank Volume"] = data["Tank Width"] * data["Tank Length"] * data["Tank Height"]
    data["HeightRatio"] = data["Vapour Height"] / data["Tank Height"]
    data["Superheat Margin"] = data["Liquid Temperature"] - data["Liquid Boiling Temperature"]

    data = pd.get_dummies(data, columns=["Sensor Position Side"], prefix="Side")
    data = pd.get_dummies(data, columns=["Status"], prefix="Status")

    y = None
    if not test_dataset:
        y = data["Target Pressure"]
        data = data.drop(["Target Pressure"], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    return X_scaled, y

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ================================
# Model Functions
# ================================
def train_xgboost(X_train, y_train, X_val, y_val):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, scoring='neg_mean_squared_error', random_state=42)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    pred = best_model.predict(X_val)
    return best_model, mean_squared_error(y_val, pred), r2_score(y_val, pred)

def train_svr(X_train, y_train, X_val, y_val):
    param_dist = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'kernel': ['rbf', 'linear']
    }
    model = SVR()
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, scoring='neg_mean_squared_error', random_state=42)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    pred = best_model.predict(X_val)
    return best_model, mean_squared_error(y_val, pred), r2_score(y_val, pred)

def build_residual_model(input_shape):
    def residual_block(x):
        shortcut = x
        x = layers.Dense(256, activation=None, kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization()(x)
        x = layers.Activation('mish')(x)
        x = GaussianNoise(0.1)(x)
        x = Dropout(0.3)(x)
        x = layers.Dense(256, activation=None, kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = layers.Activation('mish')(x)
        x = Dropout(0.3)(x)
        return x

    inputs = Input(shape=(input_shape,))
    x = layers.Dense(256, activation=None)(inputs)
    x = BatchNormalization()(x)
    x = layers.Activation('mish')(x)
    x = Dropout(0.3)(x)
    x = residual_block(x)
    x = residual_block(x)
    x = layers.Dense(1, activation="softplus")(x)

    model = Model(inputs, x)
    model.compile(optimizer=optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    return model

def train_neural_network(X_train, y_train, X_val, y_val):
    model = build_residual_model(X_train.shape[1])
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, callbacks=[early_stop], verbose=1)
    model.save("best.keras")
    val_loss, val_mae = model.evaluate(X_val, y_val, batch_size=512, verbose=1)
    return model, history, val_loss, val_mae

# ================================
# Run Pipeline
# ================================
train_data = pd.read_csv("train.csv")
X_scaled, y = preprocess(train_data)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb_model, xgb_mse, xgb_r2 = train_xgboost(X_train, y_train, X_val, y_val)
svr_model, svr_mse, svr_r2 = train_svr(X_train, y_train, X_val, y_val)
nn_model, history, val_loss, val_mae = train_neural_network(X_train, y_train, X_val, y_val)

# ================================
# Test Predictions
# ================================
test_data = pd.read_csv("test.csv")
y_test_df = pd.read_csv("sample_prediction.csv")
true_pressures = y_test_df["Target Pressure (bar)"]

test_scaled, _ = preprocess(test_data, test_dataset=True)
y_pred = nn_model.predict(test_scaled).flatten()
mape = mean_absolute_percentage_error(true_pressures, y_pred)

# Save predictions
output_df = pd.DataFrame({"ID": y_test_df["ID"], "Target Pressure (bar)": y_pred})
output_df.to_csv("predictions_output.csv", index=False)

# ================================
# Summary
# ================================
print("\n--- Model Performance ---")
print(f"[XGBoost] MSE: {xgb_mse:.4f}, R²: {xgb_r2:.4f}")
print(f"[SVR] MSE: {svr_mse:.4f}, R²: {svr_r2:.4f}")
print(f"[Neural Network] Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")
print(f"[Neural Network] Test MAPE: {mape:.4f}")
print("Predictions saved to 'predictions_output.csv'.")

# ================================
# Plot Training History
# ================================
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("MAE Curve")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.show()
