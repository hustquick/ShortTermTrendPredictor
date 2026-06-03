import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from features import add_dual_future_labels, get_feature_columns


class FastMtfExtraTreesModel:
    def __init__(self, up_model, down_model, feature_cols: list[str]):
        self.up_model = up_model
        self.down_model = down_model
        self.feature_cols = feature_cols

    def predict_one(self, features: pd.DataFrame, **kwargs) -> dict:
        x = features[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_up_signal = float(self.up_model.predict_proba(x)[:, 1][0])
        p_down_signal = float(self.down_model.predict_proba(x)[:, 1][0])
        score_sum = p_up_signal + p_down_signal
        p_up_relative = 0.5 if score_sum <= 1e-12 else p_up_signal / score_sum
        return {
            "predicted_direction": "no_trade",
            "up_probability": p_up_relative,
            "up_signal_probability": p_up_signal,
            "down_signal_probability": p_down_signal,
            "direction_edge": p_up_signal - p_down_signal,
            "confidence": max(p_up_signal, p_down_signal),
            "high_win_rate_signal": False,
            "is_valid_signal": False,
        }


def train_fast_mtf_extra_trees_model(
    feature_df: pd.DataFrame,
    *,
    n_estimators: int = 160,
    max_depth: int = 10,
    min_samples_leaf: int = 6,
) -> FastMtfExtraTreesModel:
    labeled = add_dual_future_labels(feature_df)
    valid_future = labeled["future_price"].notna()
    labeled.loc[valid_future, "up_label"] = (labeled.loc[valid_future, "future_return"] > 0).astype(int)
    labeled.loc[valid_future, "down_label"] = (labeled.loc[valid_future, "future_return"] <= 0).astype(int)
    feature_cols = get_feature_columns(labeled)
    train = labeled.dropna(subset=feature_cols + ["up_label", "down_label"]).copy()
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_up = train["up_label"].astype(int)
    y_down = train["down_label"].astype(int)
    if y_up.nunique() < 2 or y_down.nunique() < 2:
        raise ValueError("mtf extra trees labels must contain both classes")
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=7,
        n_jobs=-1,
    )
    up_model = ExtraTreesClassifier(**params)
    down_model = ExtraTreesClassifier(**params)
    up_model.fit(x, y_up)
    down_model.fit(x, y_down)
    return FastMtfExtraTreesModel(up_model, down_model, feature_cols)
