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

# Load the dataset
df = pd.read_csv('Temp_data.csv', header=None)

feature_names = ['Vel', 'CAT', 'AIT', 'VA', 'VT', 'PR', 't', 'AT']
df.columns = feature_names


X = df[['Vel', 'CAT', 'AIT', 'VA', 'VT', 'PR', 't']].values
y = df['AT'].values

# Normalize input and target
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Save scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Reshape input for RNN
X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

# Bin the target variable for stratification
y_binned = pd.qcut(y, q=10, labels=False)  # Bin continuous target into 10 quantiles

# Perform train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42, stratify=y_binned
)

# Define parameter grid for random search
param_grid = {
    'units': [10, 20, 50],
    'learning_rate': [0.001, 0.01, 0.1],
    'optimizer': ['SGD', 'Adam', 'RMSprop', 'Adagrad']
}
random_params = list(ParameterSampler(param_grid, n_iter=10, random_state=42))

# Callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

# Perform random search
results = []
histories = []
for idx, params in enumerate(random_params):
    optimizer_class = {'SGD': SGD, 'Adam': Adam, 'RMSprop': RMSprop, 'Adagrad': Adagrad}[params['optimizer']]
    optimizer = optimizer_class(learning_rate=params['learning_rate'])
    mae_scores = []
    history_list = []  # Store histories for this set of hyperparameters

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
        history_list.append(history.history)  # Save training history

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
    histories.append(history_list)  # Save all histories for this parameter set

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('random_search_results_with_time.csv', index=False)

# Set global font size
plt.rcParams.update({
    'font.size': 16,  # Increase this value for larger fonts
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Plot MAE vs hyperparameters
plt.figure(figsize=(12, 8))
sns.barplot(data=results_df, x='units', y='mae', hue='optimizer')
plt.title("MAE vs Units and Optimizer")
plt.xlabel("Units")
plt.ylabel("Mean Absolute Error")
plt.legend(title="Optimizer")
plt.savefig('mae_vs_units_optimizer.png')
plt.show()

# Plot learning rate vs MAE for each optimizer
plt.figure(figsize=(12, 8))
sns.lineplot(data=results_df, x='learning_rate', y='mae', hue='optimizer', marker='o')
plt.xscale('log')
plt.title("MAE vs Learning Rate for Each Optimizer")
plt.xlabel("Learning Rate (Log Scale)")
plt.ylabel("Mean Absolute Error")
plt.legend(title="Optimizer")
plt.savefig('mae_vs_learning_rate.png')
plt.show()

# Visualize training history for the best model
best_index = results_df['mae'].idxmin()
best_histories = histories[best_index]

for fold_idx, fold_history in enumerate(best_histories):
    plt.figure(figsize=(10, 6))
    plt.plot(fold_history['loss'], label='Training Loss')
    plt.plot(fold_history['val_loss'], label='Validation Loss')
    plt.title(f"Training vs Validation Loss (Fold {fold_idx + 1})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'training_validation_loss_fold_{fold_idx + 1}.png')
    plt.show()

# Train final model
best_params = results_df.loc[results_df['mae'].idxmin()]
final_optimizer = {'SGD': SGD, 'Adam': Adam, 'RMSprop': RMSprop, 'Adagrad': Adagrad}[best_params['optimizer']](
    learning_rate=best_params['learning_rate']
)

start_time = time.time()  # Measure training time for the final model
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

# Measure total end time
total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Save both training time and total time to a text file
with open("time_metrics.txt", "w") as file:
    file.write(f"Final Model Training Time: {final_training_time:.2f} seconds\n")
    file.write(f"Total Time for Full Code Execution: {total_execution_time:.2f} seconds\n")

# Test the final model
y_pred = model.predict(X_test).flatten()
y_pred_denormalized = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_actual, y_pred_denormalized)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_denormalized))
r2 = r2_score(y_test_actual, y_pred_denormalized)

print(f"Test Metrics:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual', marker='o')
plt.plot(y_pred_denormalized, label='Predicted', marker='x')
plt.title("Actual vs Predicted Values")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.show()

plt.figure(figsize=(10, 6))
errors = y_test_actual - y_pred_denormalized
plt.hist(errors, bins=30, alpha=0.7, label='Prediction Errors')
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('error_distribution.png')
plt.show()
