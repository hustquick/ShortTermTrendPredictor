# core/alpha_model.py

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from trainer import load_model, save_model, train_validation_model


@dataclass
class AlphaModelManager:
    """Owns model load, retrain cadence, and probability prediction."""

    retrain_interval_seconds: int
    model: object | None = None
    last_train_time: datetime | None = None

    def load(self) -> bool:
        self.model = load_model()
        if self.model is not None:
            self.last_train_time = datetime.now()
            return True
        return False

    def ensure_trained(self, df: pd.DataFrame) -> bool:
        if self.model is None:
            self._train(df)
            return True
        if self.last_train_time is None:
            self._train(df)
            return True
        if (datetime.now() - self.last_train_time).total_seconds() >= self.retrain_interval_seconds:
            self._train(df)
            return True
        return False

    def _train(self, df: pd.DataFrame):
        self.model = train_validation_model(df)
        save_model(self.model)
        self.last_train_time = datetime.now()

    @property
    def feature_cols(self) -> list[str]:
        if self.model is None:
            return []
        return list(self.model.feature_cols)

    def predict_one(self, features: pd.DataFrame) -> dict:
        if self.model is None:
            raise RuntimeError("alpha model is not trained")
        return self.model.predict_one(features, signal_filter=None)
