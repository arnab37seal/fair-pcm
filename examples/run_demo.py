"""
run_demo.py
-----------
End-to-end demo of FairPCM on the bundled synthetic datasets.

Usage:
    python examples/run_demo.py [dataset_name]

    dataset_name in {blobs_2group, blobs_3group, overlap}  (default: blobs_2group)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fair_pcm import FairPCM

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets"

DATASETS = {
    "blobs_2group": "synthetic_blobs_2group.csv",
    "blobs_3group": "synthetic_blobs_3group.csv",
    "overlap":      "synthetic_overlap.csv",
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "blobs_2group"
    csv_path = DATA_DIR / DATASETS[name]
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c != "S"]
    X = df[feature_cols].to_numpy(dtype=float)
    S = df["S"].to_numpy()

    n_clusters = 3 if name != "blobs_3group" else 4

    model = FairPCM(
        n_clusters=n_clusters,
        m=2.0,
        lam=1.0,
        tol=1e-4,
        max_iter=150,
        verbose=True,
        random_state=0,
    )
    model.fit(X, S)

    print("\nFinal objective :", model.objective_)
    print("Iterations run  :", model.n_iter_)

    report = model.fairness_report(S)
    print("\nGroup priors p_g:", report["group_priors"])
    print("KL per cluster  :", report["kl_per_cluster"])
    print("Soft proportions q_hat_kg:\n", report["soft_props"])

    # 2D visualisation (only meaningful for 2D feature spaces)
    if X.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="tab10", s=12, alpha=0.7)
        ax.scatter(model.centers_[:, 0], model.centers_[:, 1],
                   c="black", marker="x", s=150, linewidths=3, label="centers")
        ax.set_title(f"FairPCM clustering — {name}")
        ax.legend()
        out_path = Path(__file__).parent / f"{name}_result.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
