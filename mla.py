# ================================
# Imports
# ================================
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.svm import SVR
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    optimizers,
    regularizers,
    backend as K,
)
import keras_tuner as kt


# ================================
# Data Preprocessing
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

    # Logical constraints
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

    # Standardize and simplify "Status"
    data["Status"] = data["Status"].str.lower().str.replace(" ", "")
    data["Status"] = data["Status"].apply(
        lambda x: "superheated" if "h" in x else ("subcooled" if "c" in x else x)
    )

    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    if not test_dataset:
        data = remove_outliers_iqr(data, numeric_cols)

    # Feature engineering
    data["Liquid Boiling Temperature"] += 273.15
    data["Liquid Critical Temperature"] += 273.15
    data["Tank Volume"] = data["Tank Width"] * data["Tank Length"] * data["Tank Height"]
    data["HeightRatio"] = data["Vapour Height"] / data["Tank Height"]
    data["Superheat Margin"] = (
        data["Liquid Temperature"] - data["Liquid Boiling Temperature"]
    )

    data = pd.get_dummies(data, columns=["Sensor Position Side"], prefix="Side")
    data = pd.get_dummies(data, columns=["Status"], prefix="Status")

    y = None
    if not test_dataset:
        y = data["Target Pressure"]
        data = data.drop(["Target Pressure"], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    return X_scaled, y


# ================================
# Load and Prepare Data
# ================================
train_data = pd.read_csv("train.csv")
X_scaled, y = preprocess(train_data)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================================
# XGBoost Model
# ================================
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)
xgb_mse = mean_squared_error(y_val, xgb_pred)
xgb_r2 = r2_score(y_val, xgb_pred)

# ================================
# Support Vector Regression (SVR)
# ================================
svr_model = SVR(kernel="rbf", C=100, epsilon=0.1)
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_val)
svr_mse = mean_squared_error(y_val, svr_pred)
svr_r2 = r2_score(y_val, svr_pred)


# ================================
# Neural Network (Mish + Softplus)
# ================================
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


tf.keras.utils.get_custom_objects().update({"mish": mish})

model = models.Sequential(
    [
        layers.Dense(
            256,
            activation="mish",
            input_shape=(X_train.shape[1],),
            kernel_regularizer=regularizers.l2(1e-5),
        ),
        layers.Dropout(0.1),
        layers.Dense(256, activation="mish", kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.1),
        layers.Dense(256, activation="mish", kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.1),
        layers.Dense(1, activation="softplus"),
    ]
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=100, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

# ================================
# Save & Evaluate Neural Net
# ================================
model.save("best.keras")

val_loss, val_mae = model.evaluate(X_val, y_val, batch_size=512, verbose=1)


# ================================
# Test Set Prediction & MAPE
# ================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


test_data = pd.read_csv("test.csv")
y_test_df = pd.read_csv("sample_prediction.csv")
true_pressures = y_test_df["Target Pressure (bar)"]

test_scaled, _ = preprocess(test_data, test_dataset=True)
y_pred = model.predict(test_scaled).flatten()
mape = mean_absolute_percentage_error(true_pressures, y_pred)

# Save predictions
output_df = pd.DataFrame({"ID": y_test_df["ID"], "Target Pressure (bar)": y_pred})
output_df.to_csv("predictions_output.csv", index=False)

# ================================
# Metrics Summary
# ================================
print("\n--- Model Performance ---")
print(f"[XGBoost] MSE: {xgb_mse:.4f}, R²: {xgb_r2:.4f}")
print(f"[SVR] MSE: {svr_mse:.4f}, R²: {svr_r2:.4f}")
print(
    f"[Neural Network] Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}"
)
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
