"""
Phase 4 — model definitions.

Two models per the spec: (1) LogisticRegression as an interpretable
baseline, (2) a gradient-boosted tree (LightGBM, falling back to
XGBoost if LightGBM is unavailable/misbehaves) as the main model.
Both are re-fit from scratch on every walk-forward window's purged/
embargoed training slice — no state carried across windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier

    _GBM_AVAILABLE = "lightgbm"
except ImportError:  # pragma: no cover
    try:
        from xgboost import XGBClassifier

        _GBM_AVAILABLE = "xgboost"
    except ImportError:
        _GBM_AVAILABLE = None


def build_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_gbm():
    if _GBM_AVAILABLE == "lightgbm":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
    elif _GBM_AVAILABLE == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        raise ImportError("Neither lightgbm nor xgboost is installed.")


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "gbm": build_gbm,
}


def fit_predict_proba(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
):
    model = MODEL_BUILDERS[model_name]()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, np.asarray(proba)


def gbm_library_name() -> str:
    if _GBM_AVAILABLE is None:
        raise ImportError("Neither lightgbm nor xgboost is installed.")
    return _GBM_AVAILABLE
