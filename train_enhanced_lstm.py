
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)

LOOKBACK = 168
HORIZON = 24

def calendar_features(times):
    """Four cyclic calendar features for each timestamp."""
    hour = times.hour.to_numpy()
    day = times.dayofyear.to_numpy()

    return np.stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * (day - 1) / 365.25),
        np.cos(2 * np.pi * (day - 1) / 365.25)
    ], axis=-1).astype(np.float32)

def load_split(split):
    X = np.load(DATA / f"X_{split}.npy")
    Y = np.load(DATA / f"y_{split}.npy")

    starts = pd.read_csv(
        DATA / f"timestamps_{split}.csv",
        parse_dates=["forecast_start"]
    )["forecast_start"]

    # Calendar features for each historical input hour.
    past_cal = []
    future_cal = []

    for start in starts:
        past_times = pd.date_range(
            start - pd.Timedelta(hours=LOOKBACK),
            periods=LOOKBACK,
            freq="h"
        )
        future_times = pd.date_range(
            start,
            periods=HORIZON,
            freq="h"
        )

        past_cal.append(calendar_features(past_times))
        future_cal.append(calendar_features(future_times))

    past_cal = np.asarray(past_cal, dtype=np.float32)
    future_cal = np.asarray(future_cal, dtype=np.float32)

    # Historical input: PV + 4 calendar features.
    X_enhanced = np.concatenate([X, past_cal], axis=-1)

    # Future calendar features are known at prediction time.
    future_flat = future_cal.reshape(len(Y), -1)

    return X_enhanced, future_flat, Y

print("Loading enhanced training data...")
X_train, F_train, y_train = load_split("train")
X_val, F_val, y_val = load_split("val")

print("Historical input:", X_train.shape)
print("Future calendar:", F_train.shape)
print("Targets:", y_train.shape)

# Two-input neural network.
history_input = layers.Input(
    shape=(LOOKBACK, 5),
    name="historical_input"
)

future_input = layers.Input(
    shape=(HORIZON * 4,),
    name="future_calendar_input"
)

x = layers.LSTM(64)(history_input)
x = layers.Concatenate()([x, future_input])
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
output = layers.Dense(HORIZON)(x)

model = Model(
    inputs=[history_input, future_input],
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
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
    MODELS / "enhanced_lstm_best.keras",
    monitor="val_loss",
    save_best_only=True
)

print("\nTraining enhanced LSTM...")

history = model.fit(
    [X_train, F_train],
    y_train,
    validation_data=([X_val, F_val], y_val),
    epochs=35,
    batch_size=128,
    callbacks=[early_stop, checkpoint],
    shuffle=False,
    verbose=1
)

model.save(MODELS / "enhanced_lstm_final.keras")

print("\nEnhanced LSTM training completed!")
print("Best validation loss:", min(history.history["val_loss"]))
