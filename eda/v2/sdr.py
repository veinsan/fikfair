"""Sufficient Dimension Reduction helpers (SIR + SAVE). Fit INSIDE CV folds only: they use the target."""
import numpy as np


def _whiten(X):
    mu = X.mean(0)
    ev, V = np.linalg.eigh(np.cov(X.T))
    W = V @ np.diag(ev ** -0.5) @ V.T
    return (X - mu) @ W, W, mu


def sir_save(X, y, H=10, method="sir"):
    Z, W, mu = _whiten(X)
    p = X.shape[1]
    sl = np.array_split(np.argsort(y), H)
    if method == "sir":
        M = sum(len(s) / len(y) * np.outer(Z[s].mean(0), Z[s].mean(0)) for s in sl)
    else:
        M = sum(len(s) / len(y) * (np.eye(p) - np.cov(Z[s].T)) @ (np.eye(p) - np.cov(Z[s].T)) for s in sl)
    ev, V = np.linalg.eigh(M)
    return ev[::-1], W @ V[:, ::-1], mu


class SDR:
    """Project onto top-d SIR and top-d SAVE directions (sklearn-style transformer)."""
    def __init__(self, d_sir=3, d_save=3, H=10, keep_raw=False):
        self.d_sir, self.d_save, self.H, self.keep_raw = d_sir, d_save, H, keep_raw

    def fit(self, X, y):
        B = []
        if self.d_sir:
            B.append(sir_save(X, y, self.H, "sir")[1][:, : self.d_sir])
        if self.d_save:
            B.append(sir_save(X, y, self.H, "save")[1][:, : self.d_save])
        self.B = np.hstack(B)
        self.mu = X.mean(0)
        P = (X - self.mu) @ self.B
        self.sd = P.std(0)
        return self

    def transform(self, X):
        P = (X - self.mu) @ self.B / self.sd
        return np.hstack([X, P]) if self.keep_raw else P

    def get_params(self, deep=True):
        return {"d_sir": self.d_sir, "d_save": self.d_save, "H": self.H, "keep_raw": self.keep_raw}

    def set_params(self, **k):
        for a, b in k.items():
            setattr(self, a, b)
        return self
