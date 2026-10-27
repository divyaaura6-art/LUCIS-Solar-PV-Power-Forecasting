
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)

X_train = np.load(DATA / "X_train.npy")
y_train = np.load(DATA / "y_train.npy")

X_val = np.load(DATA / "X_val.npy")
y_val = np.load(DATA / "y_val.npy")

print("Training samples:", X_train.shape)
print("Validation samples:", X_val.shape)

model = models.Sequential([
    layers.Input(shape=(168, 1)),
    layers.LSTM(32),
    layers.Dense(32, activation="relu"),
    layers.Dense(24, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    MODEL_DIR / "lstm_best.keras",
    monitor="val_loss",
    save_best_only=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

model.save(MODEL_DIR / "lstm_final.keras")

print("LSTM training complete!")
print("Best validation loss:", min(history.history["val_loss"]))
print("Saved models to:", MODEL_DIR)
