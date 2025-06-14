import tensorflow as tf
import mlflow
import pandas as pd
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import dagshub

# load dataset
x_train_dataset = pd.read_csv("preprocessed-dataset/x_train_preprocessing.csv", index_col='Date').astype(np.float32)
x_valid_dataset = pd.read_csv("preprocessed-dataset/x_valid_preprocessing.csv", index_col='Date').astype(np.float32)

N_FEATURES = x_train_dataset.shape[1]
N_PAST = 60
N_FUTURE = 14
SHIFT = 1
BATCH_SIZE = 32

def windowed_dataset(series, batch_size, n_past, n_future, shift):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

train_set = windowed_dataset(x_train_dataset, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)
valid_set = windowed_dataset(x_valid_dataset, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)

# model
def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_PAST, N_FEATURES)),
        tf.keras.layers.LSTM(units=hp.Int('lstm_units', 32, 128, step=32)),
        tf.keras.layers.Dense(N_FUTURE * N_FEATURES),
        tf.keras.layers.Reshape((N_FUTURE, N_FEATURES))
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='mae',
        metrics=['mae']
    )
    return model

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=3, restore_best_weights=True)

# MLflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("/prediksi-penjualan-telur")
dagshub.init(repo_owner='dinanabila', repo_name='prediksi-penjualan-telur', mlflow=True)

tuner = kt.RandomSearch(
    build_model,
    objective="val_mae",
    max_trials=5,
    directory="tuner_logs",
    project_name="manual_tuning"
)

with mlflow.start_run():

    # tuning hyperparameter
    tuner.search(train_set, validation_data=valid_set, epochs=20, callbacks=[early_stop], verbose=1)

    best_model = tuner.get_best_models(num_models=1)[0]

    # manual logging
    val_loss, val_mae = best_model.evaluate(valid_set, verbose=0)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_mae", val_mae)
    for x_batch, y_batch in train_set.take(1):
        train_loss, train_mae = best_model.evaluate(x_batch, y_batch, verbose=0)
        mlflow.log_metric("loss", train_loss)
        mlflow.log_metric("mae", train_mae)
        break

    # prediksi
    y_preds = best_model.predict(valid_set)
    y_preds_flat = y_preds.reshape(-1, N_FEATURES)
    y_true = np.concatenate([y for _, y in valid_set], axis=0).reshape(-1, N_FEATURES)

    # tambahan metrik: RMSE dan MAPE
    rmse = np.sqrt(mean_squared_error(y_true, y_preds_flat))
    mape = mean_absolute_percentage_error(y_true, y_preds_flat)
    mlflow.log_metric("val_rmse", rmse)
    mlflow.log_metric("val_mape", mape)

    # artefak 1: simpan prediksi vs aktual ke csv
    df_pred = pd.DataFrame({f"pred_feature_{i}": y_preds_flat[:, i] for i in range(N_FEATURES)})
    df_true = pd.DataFrame({f"true_feature_{i}": y_true[:, i] for i in range(N_FEATURES)})
    df_all = pd.concat([df_true, df_pred], axis=1)
    df_all.to_csv("prediction_vs_actual.csv", index=False)
    mlflow.log_artifact("prediction_vs_actual.csv")

    # artefak 2: plot hasil prediksi
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:100, 0], label="Actual")
    plt.plot(y_preds_flat[:100, 0], label="Predicted")
    plt.title("Prediction vs Actual (First Feature)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_prediction_vs_actual.png")
    mlflow.log_artifact("plot_prediction_vs_actual.png")

    # simpan model
    best_model.save("best_model_lstm.keras")
    mlflow.log_artifact("best_model_lstm.keras")

    mlflow.keras.log_model(model=best_model, artifact_path="model")
