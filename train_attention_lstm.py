
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
MODELS = BASE / "models"
OUT = BASE / "outputs"

MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)

print("Loading real solar PV data...")

X_train = np.load(DATA / "X_train.npy")
y_train = np.load(DATA / "y_train.npy")

X_val = np.load(DATA / "X_val.npy")
y_val = np.load(DATA / "y_val.npy")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# Historical sequence input
inputs = layers.Input(shape=(168, 1))

# LSTM produces an output for every historical hour
sequence = layers.LSTM(
    32,
    return_sequences=True
)(inputs)

# Learn one attention score per historical time step
scores = layers.Dense(1)(sequence)

# Convert scores to attention weights over 168 hours
weights = layers.Softmax(axis=1, name="attention_weights")(scores)

# Weighted sum of historical LSTM representations
context = layers.Dot(axes=(2, 1))(
    [layers.Permute((2, 1))(weights), sequence]
)
context = layers.GlobalAveragePooling1D()(context)

# Predict 24 future hourly values
x = layers.Dense(32, activation="relu")(context)
outputs = layers.Dense(24)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    MODELS / "attention_lstm_best.keras",
    monitor="val_loss",
    save_best_only=True
)

print("\nTraining Attention-LSTM...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=128,
    callbacks=[early_stop, checkpoint],
    shuffle=False,
    verbose=1
)

model.save(MODELS / "attention_lstm_final.keras")

pd.DataFrame(history.history).to_csv(
    OUT / "attention_lstm_training_history.csv",
    index=False
)

print("\nAttention-LSTM training completed!")
print("Best validation loss:", min(history.history["val_loss"]))
print("Best epoch:", np.argmin(history.history["val_loss"]) + 1)
print("Saved best model:", MODELS / "attention_lstm_best.keras")
