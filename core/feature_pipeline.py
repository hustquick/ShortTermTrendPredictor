# core/feature_pipeline.py

import pandas as pd

from features import build_features


class FeaturePipeline:
    """Transforms raw candles into model-ready features."""

    def build(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
        feature_df = build_features(df)
        if feature_cols is None:
            return feature_df
        return feature_df.dropna(subset=feature_cols).copy()
