import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("embedding/bias_variance_summary.csv")
df = pd.read_csv(csv_path)

variant_order = ["paraphrasing", "light_noise", "moderate_noise", "heavy_noise", "requirements"]
df = df[df["variant"].isin(variant_order)]
df = df.dropna(subset=["dataset", "variant", "bias_mag_cos", "rms_cos"])

datasets = sorted(df["dataset"].dropna().unique())
variant_styles = {
    "paraphrasing": {"marker": "o", "color": "#1f77b4"},
    "light_noise": {"marker": "s", "color": "#9467bd"}, 
    "moderate_noise": {"marker": "D", "color": "#9467bd"},
    "heavy_noise": {"marker": "^", "color": "#9467bd"},
    "requirements": {"marker": "X", "color": "#ff7f0e"},
}

x = df["bias_mag_cos"].to_numpy()
y = df["rms_cos"].to_numpy()
x_pad = (x.max() - x.min()) * 0.08 if x.max() > x.min() else 0.05
y_pad = (y.max() - y.min()) * 0.08 if y.max() > y.min() else 0.05
xlim = (max(0.0, x.min() - x_pad), x.max() + x_pad)
ylim = (max(0.0, y.min() - y_pad), y.max() + y_pad)

plt.figure(figsize=(7.2, 6.2), dpi=140)
for var in variant_order:
    dsv = df[df["variant"] == var]
    if dsv.empty: 
        continue
    style = variant_styles[var]
    plt.scatter(
        dsv["bias_mag_cos"], dsv["rms_cos"],
        c=style["color"], marker=style["marker"], s=70,
        edgecolors="black", linewidths=0.6, alpha=0.85, label=var
    )

plt.legend(title="variant", loc="best")

plt.xlabel("bias_mag_cos")
plt.ylabel("rms_cos = sqrt(var_cos)")
plt.title("Bias–Variance Overview (all datasets)")
plt.xlim(*xlim); plt.ylim(*ylim)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("visual/bias_variance_overview.png", bbox_inches="tight")

for ds in datasets:
    dsd = df[df["dataset"] == ds]
    plt.figure(figsize=(6.6, 5.6), dpi=140)
    for var in variant_order:
        dsv = dsd[dsd["variant"] == var]
        if dsv.empty:
            continue
        style = variant_styles[var]
        plt.scatter(
            dsv["bias_mag_cos"], dsv["rms_cos"],
            c=style["color"], marker=style["marker"], s=80,
            edgecolors="black", linewidths=0.6, alpha=0.9, label=var
        )

    plt.xlabel("bias_mag_cos")
    plt.ylabel("rms_cos")
    plt.title(f"Bias-Variance: {ds}")
    plt.xlim(*xlim); plt.ylim(*ylim)
    plt.grid(alpha=0.25)
    plt.legend(title="variant", loc="best")
    plt.tight_layout()
    plt.savefig(f"visual/bias_variance_{ds}.png", bbox_inches="tight")

print("Generated: visual/bias_variance_overview.png and one PNG for each dataset.")