import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_branch_gates(X: pd.DataFrame):
    """
    Fit simple SE-style branch gates from training data.
    Gate for each sample/branch is sigmoid((mean_abs(branch_features) - mu) / sigma).
    """
    branch_cols = {}
    for c in X.columns:
        if c.startswith("b1_"):
            branch_cols.setdefault("b1", []).append(c)
        elif c.startswith("b2a_"):
            branch_cols.setdefault("b2a", []).append(c)
        elif c.startswith("b2b_"):
            branch_cols.setdefault("b2b", []).append(c)
        elif c.startswith("b3_"):
            branch_cols.setdefault("b3", []).append(c)
        elif c.startswith("b4_"):
            branch_cols.setdefault("b4", []).append(c)

    params = {"stats": {}, "branch_cols": branch_cols}
    for b, cols in branch_cols.items():
        act = X[cols].abs().mean(axis=1).to_numpy(dtype=np.float32)
        mu = float(np.mean(act))
        sigma = float(np.std(act) + 1e-8)
        params["stats"][b] = {"mu": mu, "sigma": sigma}

    return params


def apply_branch_gates(X: pd.DataFrame, params):
    out = X.copy()
    for b, cols in params["branch_cols"].items():
        if not cols:
            continue
        mu = params["stats"][b]["mu"]
        sigma = params["stats"][b]["sigma"]
        act = out[cols].abs().mean(axis=1).to_numpy(dtype=np.float32)
        g = _sigmoid((act - mu) / sigma).astype(np.float32)
        out.loc[:, cols] = out[cols].to_numpy(dtype=np.float32) * g[:, None]
    return out
