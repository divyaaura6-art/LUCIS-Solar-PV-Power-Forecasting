
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "outputs"

existing = pd.read_csv(
    OUT / "final_model_comparison.csv"
)

remaining = pd.read_csv(
    OUT / "remaining_lstm_comparison.csv"
)

remaining = remaining.rename(columns={
    "Daytime_MAE_MW": "Daylight_MAE_MW",
    "Daytime_RMSE_MW": "Daylight_RMSE_MW",
    "Daytime_R2": "Daylight_R2"
})

print("Existing columns:", existing.columns.tolist())
print("New columns:", remaining.columns.tolist())

# Ensure identical column names
assert set(existing.columns) == set(remaining.columns), (
    "Column names differ. Inspect the printed columns "
    "before combining the results."
)

combined = pd.concat(
    [existing, remaining],
    ignore_index=True
)

assert len(combined) == 6
assert combined["Model"].is_unique

combined = combined.sort_values(
    "RMSE_MW"
).reset_index(drop=True)

combined.to_csv(
    OUT / "final_six_model_comparison.csv",
    index=False
)

for metric, title, filename in [
    ("MAE_MW", "Model Comparison — MAE", "final_six_mae.png"),
    ("RMSE_MW", "Model Comparison — RMSE", "final_six_rmse.png"),
    ("R2", "Model Comparison — R²", "final_six_r2.png"),
    (
        "Daylight_RMSE_MW",
        "Daytime Model Comparison — RMSE",
        "final_six_daytime_rmse.png"
    )
]:
    plot_data = combined.sort_values(
        metric,
        ascending=(metric != "R2")
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        plot_data["Model"],
        plot_data[metric]
    )

    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("\nFINAL SIX-MODEL COMPARISON")
print(combined.to_string(index=False))
print("\nSaved final CSV and four comparison graphs.")
