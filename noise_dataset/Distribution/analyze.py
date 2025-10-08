import os
import re
import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple

EMB_DIR = "embedding"

DATASETS   = ["drop", "hotpotqa", "math", "gsm8k", "humaneval", "mbpp"]
VARIANTS   = ["original", "requirements", "paraphrasing",
              "light_noise", "moderate_noise", "heavy_noise"]

def l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n

def bias_variance(D: np.ndarray) -> Tuple[float, float, np.ndarray]:
    b = D.mean(axis=0)
    R = D - b
    var = np.mean(np.sum(R * R, axis=1))
    return float(np.linalg.norm(b)), float(var), b

def radial_angular_stats(Ou: np.ndarray, Mu: np.ndarray) -> Dict[str, float]:
    D = Mu - Ou
    s = np.sum(D * Ou, axis=1)
    P = D - s[:, None] * Ou
    cos_sim = np.clip(np.sum(Ou * Mu, axis=1), -1.0, 1.0)
    theta = np.arccos(cos_sim)

    return {
        "rad_bias": float(s.mean()),
        "rad_std":  float(s.std()),
        "perp_mean": float(np.linalg.norm(P, axis=1).mean()),
        "perp_std":  float(np.linalg.norm(P, axis=1).std()),
        "angle_mean_rad": float(theta.mean()),
        "angle_std_rad":  float(theta.std()),
        "angle_mean_deg": float(theta.mean() * 180.0 / math.pi),
        "angle_std_deg":  float(theta.std()  * 180.0 / math.pi),
    }

def length_change_stats(O: np.ndarray, M: np.ndarray) -> Dict[str, float]:
    rO = np.linalg.norm(O, axis=1)
    rM = np.linalg.norm(M, axis=1)
    dR = rM - rO
    return {
        "delta_norm_mean": float(dR.mean()),
        "delta_norm_std":  float(dR.std()),
        "orig_norm_mean":  float(rO.mean()),
        "mod_norm_mean":   float(rM.mean()),
    }

def load_embeddings_for_dataset(ds: str) -> Dict[str, np.ndarray]:
    out = {}
    for var in VARIANTS:
        path = os.path.join(EMB_DIR, f"{ds}_{var}_embeddings.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path} should be 2D, got {arr.shape}")
        out[var] = arr
    shapes = {k: v.shape for k, v in out.items()}
    n_set = {v.shape[0] for v in out.values()}
    d_set = {v.shape[1] for v in out.values()}
    if len(n_set) != 1 or len(d_set) != 1:
        raise ValueError(f"{ds} variant shapes are inconsistent: {shapes}")
    return out

rows = []
for ds in DATASETS:
    embs = load_embeddings_for_dataset(ds)
    O_raw = embs["original"]
    Ou    = l2n(O_raw)

    for var in VARIANTS:
        if var == "original":
            continue
        M_raw = embs[var]
        Mu    = l2n(M_raw)

        D_cos = Mu - Ou
        bias_mag_cos, var_cos, b_cos = bias_variance(D_cos)
        rms_cos = math.sqrt(var_cos)

        ra = radial_angular_stats(Ou, Mu)

        D_raw = M_raw - O_raw
        bias_mag_raw, var_raw, b_raw = bias_variance(D_raw)
        rms_raw = math.sqrt(var_raw)

        ln_stats = length_change_stats(O_raw, M_raw)

        rows.append({
            "dataset": ds,
            "variant": var,
            "N": O_raw.shape[0],
            "d": O_raw.shape[1],
            "bias_mag_cos": bias_mag_cos,
            "var_cos": var_cos,
            "rms_cos": rms_cos,
            "rad_bias": ra["rad_bias"],
            "rad_std":  ra["rad_std"],
            "perp_mean": ra["perp_mean"],
            "perp_std":  ra["perp_std"],
            "angle_mean_deg": ra["angle_mean_deg"],
            "angle_std_deg":  ra["angle_std_deg"],
            "bias_mag_raw": bias_mag_raw,
            "var_raw": var_raw,
            "rms_raw": rms_raw,
            **ln_stats,
        })

df = pd.DataFrame(rows).sort_values(["dataset","variant"]).reset_index(drop=True)

out_path = os.path.join(EMB_DIR, "bias_variance_summary.csv")
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"[OK] Saved: {out_path}")
with pd.option_context('display.max_columns', None, 'display.width', 160):
    print(df.head(12))