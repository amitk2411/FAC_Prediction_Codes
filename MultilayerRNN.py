# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:13:48 2024

@author: AK
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import random
import time

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

total_start_time = time.time()

# Loading dataset
df = pd.read_csv('Temp_data.csv', header=None)

feature_names = ['Vel', 'CAT', 'AIT', 'VA', 'VT', 'PR', 't', 'AT']
df.columns = feature_names

X = df[['Vel', 'CAT', 'AIT', 'VA', 'VT', 'PR', 't']].values
y = df['AT'].values

# Normalizing
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Saving scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Reshaping input for RNN
X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

# stratification bins
y_binned = pd.qcut(y, q=10, labels=False)  # Bin continuous target into 10 quantiles

# Performing train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42, stratify=y_binned
)

# Defining parameter grid for random search
param_grid = {
    'units': [10, 20, 50],
    'learning_rate': [0.001, 0.01, 0.1],
    'optimizer': ['SGD', 'Adam', 'RMSprop', 'Adagrad']
}
random_params = list(ParameterSampler(param_grid, n_iter=10, random_state=42))

# Callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

# Performing random search
results = []
histories = []
for idx, params in enumerate(random_params):
    optimizer_class = {'SGD': SGD, 'Adam': Adam, 'RMSprop': RMSprop, 'Adagrad': Adagrad}[params['optimizer']]
    optimizer = optimizer_class(learning_rate=params['learning_rate'])
    mae_scores = []
    history_list = []

    start_time = time.time()
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model = Sequential([
            SimpleRNN(units=params['units'], activation='tanh', input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=100, batch_size=32, verbose=0,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping]
        )
        history_list.append(history.history)

        y_val_pred = model.predict(X_val_fold).flatten()
        y_val_denormalized = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
        y_val_actual = scaler_y.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()
        mae_scores.append(mean_absolute_error(y_val_actual, y_val_denormalized))

    elapsed_time = time.time() - start_time
    mean_mae = np.mean(mae_scores)

    results.append({
        'optimizer': params['optimizer'],
        'units': params['units'],
        'learning_rate': params['learning_rate'],
        'mae': mean_mae,
        'training_time': elapsed_time
    })
    histories.append(history_list)

# Converting results to a DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('random_search_results_with_time.csv', index=False)

# Training final model
best_params = results_df.loc[results_df['mae'].idxmin()]
final_optimizer = {'SGD': SGD, 'Adam': Adam, 'RMSprop': RMSprop, 'Adagrad': Adagrad}[best_params['optimizer']](
    learning_rate=best_params['learning_rate']
)

start_time = time.time()
model = Sequential([
    SimpleRNN(units=int(best_params['units']), activation='tanh', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer=final_optimizer, loss='mean_squared_error')

history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=32, validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
final_training_time = time.time() - start_time
model.save('final_model_MultilayerRNN.h5')

# Testing the final model
y_pred = model.predict(X_test).flatten()
y_pred_denormalized = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_actual, y_pred_denormalized)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_denormalized))
r2 = r2_score(y_test_actual, y_pred_denormalized)

# Bootstrapping for confidence intervals
n_bootstraps = 1000
bootstrap_mae = []
bootstrap_rmse = []
bootstrap_r2 = []

np.random.seed(42)
for i in range(n_bootstraps):
    indices = np.random.choice(range(len(y_test)), size=len(y_test), replace=True)
    y_test_sample = y_test[indices]
    y_pred_sample = y_pred[indices]

    y_test_sample_denorm = scaler_y.inverse_transform(y_test_sample.reshape(-1, 1)).flatten()
    y_pred_sample_denorm = scaler_y.inverse_transform(y_pred_sample.reshape(-1, 1)).flatten()

    bootstrap_mae.append(mean_absolute_error(y_test_sample_denorm, y_pred_sample_denorm))
    bootstrap_rmse.append(np.sqrt(mean_squared_error(y_test_sample_denorm, y_pred_sample_denorm)))
    bootstrap_r2.append(r2_score(y_test_sample_denorm, y_pred_sample_denorm))

mae_lower, mae_upper = np.percentile(bootstrap_mae, [2.5, 97.5])
rmse_lower, rmse_upper = np.percentile(bootstrap_rmse, [2.5, 97.5])
r2_lower, r2_upper = np.percentile(bootstrap_r2, [2.5, 97.5])

print(f"Test Metrics:\nMAE: {mae:.4f} with 95% CI: ({mae_lower:.4f}, {mae_upper:.4f})")
print(f"RMSE: {rmse:.4f} with 95% CI: ({rmse_lower:.4f}, {rmse_upper:.4f})")
print(f"R²: {r2:.4f} with 95% CI: ({r2_lower:.4f}, {r2_upper:.4f})")

# Visualizing confidence intervals
metrics = ['MAE', 'RMSE', 'R²']
means = [mae, rmse, r2]
lower_bounds = [mae_lower, rmse_lower, r2_lower]
upper_bounds = [mae_upper, rmse_upper, r2_upper]

plt.figure(figsize=(10, 6))
plt.bar(metrics, means, yerr=[np.array(means) - np.array(lower_bounds), np.array(upper_bounds) - np.array(means)],
        capsize=10, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Error Metrics with 95% Confidence Intervals")
plt.ylabel("Metric Value")
plt.savefig('metrics_with_confidence_intervals.png')
plt.show()
