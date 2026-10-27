
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "nhits_data"
MODELS = BASE / "models"
OUT = BASE / "outputs"

MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

print("Loading N-HiTS training data...")

train = pd.read_csv(DATA / "train.csv", parse_dates=["ds"])
val = pd.read_csv(DATA / "val.csv", parse_dates=["ds"])

print("Training hours:", len(train))
print("Validation hours:", len(val))

# Training and validation are contiguous.
# NeuralForecast uses the final val_size observations
# for validation and excludes them from training targets.
train_val = pd.concat(
    [train, val],
    ignore_index=True
)

model = NHITS(
    h=24,
    input_size=168,
    loss=MSE(),
    max_steps=1000,
    val_check_steps=100,
    early_stop_patience_steps=5,
    batch_size=32,
    windows_batch_size=256,
    learning_rate=0.001,
    scaler_type="identity",
    random_seed=42,
    accelerator="cpu",
    devices=1,
    enable_progress_bar=True
)

nf = NeuralForecast(
    models=[model],
    freq="h"
)

print("\nTraining N-HiTS...")
print("Validation size:", len(val))

nf.fit(
    df=train_val,
    val_size=len(val)
)

nf.save(
    path=str(MODELS / "nhits"),
    overwrite=True,
    save_dataset=False
)

print("\nN-HiTS training completed!")
print("Saved model:", MODELS / "nhits")
