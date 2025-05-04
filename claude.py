import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers
import optuna
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.optimizers import AdamW, SGD
from tensorflow.keras.activations  import mish
import shap


# Enable mixed precision for faster training on compatible GPUs
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
except:
    print("Mixed precision not available")

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Display settings
pd.set_option('display.max_columns', None)

# Improved outlier removal with adjustable IQR multiplier
def remove_outliers_iqr(df, features, multiplier=1.5):
    """
    Remove outliers based on IQR method with adjustable multiplier
    Returns both cleaned dataframe and identified outliers
    """
    df_clean = df.copy()
    outliers_mask = np.zeros(len(df), dtype=bool)
    
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        col_outliers = ~((df[col] >= lower) & (df[col] <= upper))
        outliers_mask = outliers_mask | col_outliers
        
    df_clean = df[~outliers_mask]
    df_outliers = df[outliers_mask]
    
    print(f"Removed {len(df_outliers)} outliers ({len(df_outliers)/len(df)*100:.2f}%)")
    return df_clean, df_outliers

# Enhanced feature engineering
def engineer_features(data):
    """Extended feature engineering with physical relationships"""
    # Basic volume and ratio calculations
    data["Tank Volume"] = data["Tank Width"] * data["Tank Length"] * data["Tank Height"]
    data["HeightRatio"] = data["Vapour Height"] / data["Tank Height"]
    data["Liquid Height"] = data["Tank Height"] - data["Vapour Height"]
    data["Liquid Volume"] = data["Tank Volume"] * (1 - data["HeightRatio"])
    data["Vapour Volume"] = data["Tank Volume"] * data["HeightRatio"]
    
    # Temperature-related features
    data["Superheat Margin"] = data["Liquid Temperature"] - data["Liquid Boiling Temperature"]
    data["Temp Ratio"] = data["Vapour Temperature"] / data["Liquid Temperature"]
    data["Critical Temp Ratio"] = data["Liquid Temperature"] / data["Liquid Critical Temperature"]
    
    # Dimensionless numbers and ratios
    data["Width_Length_Ratio"] = data["Tank Width"] / data["Tank Length"]
    data["Height_Width_Ratio"] = data["Tank Height"] / data["Tank Width"]
    data["Surface_Area"] = 2 * (data["Tank Width"] * data["Tank Length"] + 
                               data["Tank Height"] * data["Tank Length"] + 
                               data["Tank Height"] * data["Tank Width"])
    data["Volume_Surface_Ratio"] = data["Tank Volume"] / data["Surface_Area"]
    
    # Polynomial and interaction features
    data["HeightRatio_Squared"] = data["HeightRatio"] ** 2
    data["Superheat_Squared"] = data["Superheat Margin"] ** 2
    data["Height_Temp_Interaction"] = data["HeightRatio"] * data["Superheat Margin"]
    
    # Log transforms for skewed features
    for col in ["Tank Volume", "Liquid Volume", "Vapour Volume"]:
        data[f"Log_{col}"] = np.log1p(data[col])
    
    return data

def preprocess(data, test_dataset=False, scaler=None, feature_selector=None):
    """Enhanced preprocessing pipeline with better feature engineering and handling"""
    # Make a copy to avoid modifying the original
    data = data.copy()
    
    # Clean column names
    data.columns = [re.sub(r'\s*\([^)]*\)', '', col).strip() for col in data.columns]
    
    # Drop unnecessary columns
    if "Unnamed: 0" in data.columns:
        data = data.drop(["Unnamed: 0"], axis="columns")
    
    # Save target if present before any filtering
    y = None
    if "Target Pressure" in data.columns and not test_dataset:
        y = data["Target Pressure"].copy()
        data = data.drop(["Target Pressure"], axis="columns")
    
    # Basic cleaning - only drop rows with all NaN
    data = data.replace([np.inf, -np.inf], np.nan)
    missing_rows = data.isna().all(axis=1)
    data = data[~missing_rows]
    if y is not None:
        y = y[~missing_rows]
    
    # Handle missing values with advanced imputation
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Impute missing values (separately for numeric and categorical)
    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])
    
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
    
    # Apply logical constraints - keep track of filtered rows
    constraint_mask = (
        (data['Tank Width'] > 0) &
        (data['Tank Length'] > 0) &
        (data['Tank Height'] > 0) &
        (data['Vapour Height'] >= 0) &
        (data['Vapour Temperature'] > 0) &
        (data['Liquid Temperature'] > 0)
    )
    data = data[constraint_mask]
    if y is not None:
        y = y[constraint_mask]
    
    # Remove duplicates for training data - keep track of indices
    if not test_dataset:
        # Get duplicate mask (True for duplicates to keep)
        duplicate_mask = ~data.duplicated()
        data = data[duplicate_mask]
        if y is not None:
            y = y[duplicate_mask]
    
    # Standardize the categorical values
    if 'Status' in data.columns:
        data["Status"] = data["Status"].str.lower().str.replace(' ','')
        data['Status'] = data['Status'].apply(lambda x: 'superheated' if 'h' in x else ('subcooled' if 'c' in x else x))
    
    # Convert temperatures to absolute scale
    data["Liquid Boiling Temperature"] = data["Liquid Boiling Temperature"] + 273.15
    data["Liquid Critical Temperature"] = data["Liquid Critical Temperature"] + 273.15
    
    # Enhanced feature engineering
    data = engineer_features(data)
    
    # Remove outliers only from training data
    if not test_dataset:
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data_before = len(data)
        data, outliers = remove_outliers_iqr(data, numeric_cols, multiplier=2.0)
        # Find which rows were kept (not outliers)
        if y is not None and data_before > len(data):
            # Reset index and get the kept indices
            data_indices = data.index
            y = y.loc[data_indices]
    
    # One-hot encoding for categorical features
    if 'Sensor Position Side' in data.columns:
        data = pd.get_dummies(data, columns=["Sensor Position Side"], prefix="Side")
    if 'Status' in data.columns:
        data = pd.get_dummies(data, columns=["Status"], prefix="Status")
    
    # Feature selection - only apply to training data or use pre-fitted selector
    if not test_dataset and feature_selector is None:
        # Use L1 regularization to select important features
        from sklearn.linear_model import Lasso
        # Verify data and y have same length
        print(f"Data shape: {data.shape[0]}, Target shape: {len(y)}")
        assert data.shape[0] == len(y), "Data and target lengths don't match!"
        
        selector = SelectFromModel(Lasso(alpha=0.01, random_state=SEED), threshold="mean")
        selector.fit(data, y)
        
        # Get selected features
        selected_features = data.columns[selector.get_support()]
        print(f"Selected {len(selected_features)} out of {len(data.columns)} features")
        
        # Keep only selected features
        data = data[selected_features]
        return data, y, selector, selected_features
    
    elif test_dataset and feature_selector is not None:
        # For test data, use same features as training
        data = data[feature_selector]
    
    # Scaling - fit on training or use pre-fitted scaler
    if scaler is None:
        # Use RobustScaler for better handling of remaining outliers
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(data)
        return X_scaled, y, scaler
    else:
        X_scaled = scaler.transform(data)
        return X_scaled, y
    
    return X_scaled, y

# Model architecture with latest best practices
def create_model(input_shape, hidden_units=[256, 256, 256], activation='mish', 
                l2_reg=1e-5, dropout_rate=0.1, learning_rate=0.001, use_bn=True):
    """Create an optimized neural network model"""
    
    # Using Functional API for more flexibility
    inputs = tf.keras.Input(shape=(input_shape,))
    
    # Add Gaussian noise for better regularization
    x = GaussianNoise(0.1)(inputs)
    
    # Create hidden layers
    for i, units in enumerate(hidden_units):
        x = Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
        if use_bn:
            x = BatchNormalization()(x)
        if activation == 'mish':
            x = mish(x)
        else:
            x = layers.Activation(activation)(x)
        x = Dropout(dropout_rate)(x)
    
    # Output layer with softplus to ensure positive pressure predictions
    outputs = Dense(1, activation='softplus')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use AdamW optimizer for better generalization
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

# Hyperparameter optimization with Optuna
def objective(trial, X_train, y_train, X_val, y_val):
    # Hyperparameters to tune
    hidden_units = [
        trial.suggest_int(f'hidden_units_1', 64, 512),
        trial.suggest_int(f'hidden_units_2', 64, 512),
        trial.suggest_int(f'hidden_units_3', 64, 512)
    ]
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    
    # Create and train model
    model = create_model(
        input_shape=X_train.shape[1],
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        l2_reg=l2_reg
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=30,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Return best validation loss
    return min(history.history['val_loss'])

# Main script
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("train.csv")
    
    # Explore data
    print(f"Data shape: {data.shape}")
    print(f"Missing values: {data.isna().sum().sum()}")
    
    # Apply preprocessing with enhanced feature engineering
    X_data, y, feature_selector, selected_features = preprocess(data)
    
    # Apply scaling to training data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # Split data with stratification if applicable
    # For regression problems, we can create bins for stratification
    try:
        y_bins = pd.qcut(y, q=5, labels=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=SEED, stratify=y_bins
        )
    except ValueError:
        # If qcut fails due to duplicate values, use regular split
        print("Could not create equal-sized bins, using regular split")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=SEED
        )
    
    print("Training Set Size:", X_train.shape)
    print("Validation Set Size:", X_val.shape)
    
    # Hyperparameter optimization with Optuna
    print("Starting hyperparameter optimization...")
    
    # Check if we have a small dataset - if so, use a simpler optimization
    if len(X_train) < 1000:
        print("Small dataset detected, using simplified optimization")
        # Default hyperparameters
        best_params = {
            'hidden_units_1': 256,
            'hidden_units_2': 128,
            'hidden_units_3': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'l2_reg': 1e-5,
            'batch_size': 32
        }
    else:
        # Full optimization
        study = optuna.create_study(direction='minimize')
        # Pass the data explicitly to objective
        objective_with_data = lambda trial: objective(trial, X_train, y_train, X_val, y_val)
        study.optimize(objective_with_data, n_trials=30)
        
        print('Best trial:')
        trial = study.best_trial
        print(f'  Value: {trial.value}')
        print('  Params:')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
        
        best_params = trial.params
    
    # Train final model with best hyperparameters
    best_params = study.best_params
    
    # Create final model with best hyperparameters
    final_model = create_model(
        input_shape=X_train.shape[1],
        hidden_units=[
            best_params['hidden_units_1'],
            best_params['hidden_units_2'],
            best_params['hidden_units_3']
        ],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        l2_reg=best_params['l2_reg']
    )
    
    # Create learning rate scheduler for better convergence
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Early stopping with longer patience for final model
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint to save best model
    checkpoint = callbacks.ModelCheckpoint(
        'best_pressure_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train with k-fold cross validation for robust evaluation
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_data)):
        print(f"\nTraining fold {fold+1}/{n_folds}")
        
        # Get data for this fold
        X_fold_train, X_fold_val = X_data[train_idx], X_data[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create and train model
        fold_model = create_model(
            input_shape=X_data.shape[1],
            hidden_units=[
                best_params['hidden_units_1'],
                best_params['hidden_units_2'],
                best_params['hidden_units_3']
            ],
            dropout_rate=best_params['dropout_rate'],
            learning_rate=best_params['learning_rate'],
            l2_reg=best_params['l2_reg']
        )
        
        history = fold_model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=300,
            batch_size=best_params['batch_size'],
            callbacks=[early_stop, lr_scheduler],
            verbose=1
        )
        
        # Evaluate fold performance
        fold_val_loss = min(history.history['val_loss'])
        fold_scores.append(fold_val_loss)
        print(f"Fold {fold+1} validation loss: {fold_val_loss:.4f}")
    
    print(f"\nCross-validation results:")
    print(f"Mean validation loss: {np.mean(fold_scores):.4f}")
    print(f"Std validation loss: {np.std(fold_scores):.4f}")
    
    # Train final model on full training data
    print("\nTraining final model on all training data...")
    final_history = final_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=best_params['batch_size'],
        callbacks=[early_stop, lr_scheduler, checkpoint],
        verbose=1
    )
    
    # Evaluate final model
    final_val_loss = min(final_history.history['val_loss'])
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(final_history.history['loss'], label='Training Loss')
    plt.plot(final_history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(final_history.history['mae'], label='Training MAE')
    plt.plot(final_history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()
    
    # Feature importance analysis using permutation importance (more reliable than SHAP for this case)
    try:
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            final_model, X_val, y_val, n_repeats=10, random_state=SEED
        )
        
        # Sort features by importance
        sorted_idx = perm_importance.importances_mean.argsort()
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [selected_features[i] for i in sorted_idx])
        plt.title("Permutation Feature Importance")
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("Feature importance analysis completed and saved.")
    except Exception as e:
        print(f"Feature importance analysis failed: {e} - continuing without it.")
    
    # Save model and preprocessing information
    final_model.save('final_pressure_model.h5')
    
    # Make predictions on validation set
    y_pred = final_model.predict(X_val)
    
    # Calculate regression metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"\nFinal Model Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual Pressure')
    plt.ylabel('Predicted Pressure')
    plt.title('Predicted vs Actual Pressure')
    plt.savefig('predictions_vs_actual.png')
    plt.close()
    
    print("Optimization complete! Model saved as 'final_pressure_model.h5'")