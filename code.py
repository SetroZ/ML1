#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, regularizers, activations, callbacks, optimizers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import re
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

from tensorflow.keras.models import load_model

pd.set_option('display.max_columns', None)


# In[ ]:


def remove_outliers_iqr(df, features):
    
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df



def preprocess(data,test_dataset=False):
# Strip (k) (m) and (bar) from keys
    data.columns = [re.sub(r'\s*\([^)]*\)', '', col).strip() for col in data.columns]
    # drop missing values
    data = data.drop(["Unnamed: 0"],axis="columns")
    data = data.dropna(axis=0)
    #logical constraints
    data = data[(data['Tank Width'] > 0) & 
                (data['Tank Length'] > 0) & 
                (data['Tank Height'] > 0) & 
                (data['Vapour Height'] >= 0) &
                (data['Vapour Temperature'] > 0) &
                (data['Liquid Temperature'] > 0)]
    if(not test_dataset):
        data = data.drop_duplicates()
    data["Status"] = data["Status"].str.lower().str.replace(' ','')

    data['Status'] = data['Status'].apply(lambda x: 'superheated' if 'h' in x else ('subcooled' if 'c' in x else x))

        
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if(not test_dataset):
        data = remove_outliers_iqr(data, numeric_cols)
    data["Liquid Boiling Temperature"] = data["Liquid Boiling Temperature"] +273.15
    data["Liquid Critical Temperature"] = data["Liquid Critical Temperature"] +273.15
    data["Tank Volume"] = data["Tank Width"] * data["Tank Length"] * data["Tank Height"]  
    data["HeightRatio"]= data["Vapour Height"] / data["Tank Height"] 
    data["Superheat Margin"] = data["Liquid Temperature"] - data["Liquid Boiling Temperature"]

    data = pd.get_dummies(data, columns=["Sensor Position Side"], prefix="Side")
    data = pd.get_dummies(data, columns=["Status"], prefix="Status")
    y =  None
    if (not test_dataset):
        y= data[ "Target Pressure"] 
        data = data.drop(["Target Pressure"], axis="columns")
    print(data.head(1))
    # StandardScaler for inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    return X_scaled,y



def load_data(csvFileName):
    data = pd.read_csv(csvFileName)
    X_scaled,y = preprocess(data);
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print("Training Set Size:", X_train.shape)
    print("Validation Set Size:", X_val.shape)
    return  X_train, X_val, y_train, y_val



# In[ ]:


def train_xgboost():
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


def train_svr():
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






# In[ ]:


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def build_nn():
    model = models.Sequential([
        layers.Dense(256, activation='mish', input_shape=(X_train.shape[1],),
                    kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.1),
        layers.Dense(256, activation='mish', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.1),
        layers.Dense(256, activation='mish', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.1),
        layers.Dense(1, activation='softplus')
    ])

    optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=100, restore_best_weights=True
    )
    return model


def train_nn():
    history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
    
)






# In[ ]:


model = load_model('best.keras')







# Evaluate the model on the validation dataset only
val_loss, val_mae = model.evaluate(X_val, y_val, batch_size=512, verbose=1)

# Print the results
print(f"Validation Loss: {val_loss}")
print(f"Validation MAE: {val_mae}")


# # Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=2000,
#     batch_size=512,
#     callbacks=[early_stop],
#     verbose=1
# )


# In[10]:


model.save('best.keras')


# In[11]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Training & Validation MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training & Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()


# In[96]:


# Assuming you have X_test and y_test prepared
# 1. Evaluate the model on the test set
test_data =   pd.read_csv("test.csv")
y_test = pd.read_csv("sample_prediction.csv")
predData = y_test["Target Pressure (bar)"]
test_scaled,_ = preprocess(test_data,True)
test_loss, test_mae = model.evaluate(test_scaled, predData)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")


y_pred = model.predict(test_scaled).flatten()  # Ensure it's a 1D array

# 2. Get the true target values
y_true = y_test["Target Pressure (bar)"].values



# 3. Calculate MAPE (excluding zero targets to avoid division by zero)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# 4. Compute and print MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape/10}")


# Assuming the ID column in the test data is named 'ID'
# 1. Create a DataFrame with the predictions and IDs
output_df = pd.DataFrame({
    'ID': y_test['ID'],  # Assuming 'ID' is in the test_data
    'Target Pressure (bar)': y_pred  # The predicted values
})

# 2. Save the DataFrame to a CSV file
output_df.to_csv('predictions_output.csv', index=False)
print("Predictions have been saved to 'predictions_output.csv'.")


# In[ ]:


import keras_tuner as kt

def build_model(hp):
    model = models.Sequential()
    
    # Tunable units
    for i in range(hp.Int("num_layers", 2, 4)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=128, max_value=512, step=64),
            activation='mish',
            kernel_regularizer=regularizers.l2(hp.Choice('l2_reg', [1e-5, 1e-4, 1e-3]))
        ))
        model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(layers.Dense(1, activation='softplus'))
    
    # Compile
    optimizer = optimizers.SGD(
        learning_rate=hp.Choice('learning_rate', [0.01, 0.05, 0.1]),
        momentum=hp.Float('momentum', min_value=0.5, max_value=0.95, step=0.05)
    )
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model





tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='mish_model'
)

# Register Mish
tf.keras.utils.get_custom_objects().update({'mish': mish})

# Run search
tuner.search(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10)],
    verbose=1
)


best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]

print("Best hyperparameters:")
for key in best_hps.values:
    print(f"{key}: {best_hps.get(key)}")


