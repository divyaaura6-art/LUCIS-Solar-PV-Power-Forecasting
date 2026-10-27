
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Previously evaluated on 243 midnight-issued
# day-ahead forecasts in 2026.
# These are the recorded experimental results.
results = pd.DataFrame([
    {
        "Model": "Random Forest",
        "MAE_MW": 0.1974,
        "RMSE_MW": 0.4107,
        "R2": 0.9096,
        "Daylight_MAE_MW": 0.3639,
        "Daylight_RMSE_MW": 0.5580,
        "Daylight_R2": 0.8167
    },
    {
        "Model": "Original LSTM",
        "MAE_MW": 0.2203,
        "RMSE_MW": 0.4309,
        "R2": 0.9005,
        "Daylight_MAE_MW": 0.3909,
        "Daylight_RMSE_MW": 0.5847,
        "Daylight_R2": 0.7988
    },
    {
        "Model": "Enhanced LSTM",
        "MAE_MW": 0.2753,
        "RMSE_MW": 0.5089,
        "R2": 0.8612,
        "Daylight_MAE_MW": 0.4990,
        "Daylight_RMSE_MW": 0.6909,
        "Daylight_R2": 0.7191
    },
    {
        "Model": "N-HiTS",
        "MAE_MW": 0.2118,
        "RMSE_MW": 0.4427,
        "R2": 0.8950,
        "Daylight_MAE_MW": 0.3812,
        "Daylight_RMSE_MW": 0.6012,
        "Daylight_R2": 0.7873
    }
])

results = results.sort_values("RMSE_MW")
results.to_csv(
    OUT / "final_model_comparison.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print(results.to_string(index=False))

# Generate separate publication-ready figures
metrics = [
    ("MAE_MW", "Mean Absolute Error", "MAE (MW)",
     "model_comparison_mae.png", False),

    ("RMSE_MW", "Root Mean Squared Error", "RMSE (MW)",
     "model_comparison_rmse.png", False),

    ("R2", "Coefficient of Determination", "R²",
     "model_comparison_r2.png", True),

    ("Daylight_RMSE_MW", "Daytime RMSE Comparison",
     "Daytime RMSE (MW)",
     "model_comparison_daytime_rmse.png", False)
]

for col, title, ylabel, filename, higher_better in metrics:

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        results["Model"],
        results[col],
        color=["#2878B5", "#62A8D1", "#92BBD7", "#BACEDC"]
    )

    best_idx = (
        results[col].idxmax()
        if higher_better
        else results[col].idxmin()
    )

    for i, (idx, row) in enumerate(results.iterrows()):
        value = row[col]

        if idx == best_idx:
            bars[i].set_color("#2A9D65")

        ax.text(
            i,
            value + (0.015 if col == "R2" else 0.008),
            f"{value:.4f}",
            ha="center",
            fontsize=10
        )

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(
        0,
        results[col].max() * 1.15
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        OUT / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Saved:", filename)

print("\nAll comparison graphs generated successfully!")
