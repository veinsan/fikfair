"""Shared paths + loader for all EDA scripts. Run scripts from repo root: python eda/01_overview.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def load():
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    feats = [c for c in tr.columns if c.startswith("feature_")]
    return tr, te, feats


def save(fig, name):
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p.relative_to(ROOT)}")


def h(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
