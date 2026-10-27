
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
MODELS = BASE / "models"
OUT = BASE / "outputs"

MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)

print("Loading real PV training data...")

X_train = np.load(DATA / "X_train.npy")
y_train = np.load(DATA / "y_train.npy")

X_val = np.load(DATA / "X_val.npy")
y_val = np.load(DATA / "y_val.npy")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)

model = models.Sequential([
    layers.Input(shape=(168, 1)),

    layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling1D(pool_size=2),

    layers.LSTM(32),

    layers.Dense(32, activation="relu"),

    layers.Dense(24, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    MODELS / "cnn_lstm_best.keras",
    monitor="val_loss",
    save_best_only=True
)

print("\nTraining CNN-LSTM...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=35,
    batch_size=128,
    callbacks=[early_stop, checkpoint],
    shuffle=False,
    verbose=1
)

model.save(MODELS / "cnn_lstm_final.keras")

pd.DataFrame(history.history).to_csv(
    OUT / "cnn_lstm_training_history.csv",
    index=False
)

print("\nCNN-LSTM training completed!")
print("Best validation loss:", min(history.history["val_loss"]))
print("Best epoch:", np.argmin(history.history["val_loss"]) + 1)
print("Saved model:", MODELS / "cnn_lstm_best.keras")
