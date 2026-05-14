# kronos_adapter.py

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import PREDICT_HORIZON_MINUTES


@dataclass
class KronosForecastResult:
    available: bool
    direction: str
    confidence: float
    forecast_close: float | None
    current_close: float | None
    reason: str


def _load_kronos_components(model_name: str, tokenizer_name: str):
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    except Exception as exc:
        raise ImportError(
            "Kronos is not installed or not importable. "
            "Clone/install https://github.com/shiyu-coder/Kronos and make its model package available."
        ) from exc

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    return KronosPredictor(model, tokenizer, max_context=512)


class KronosAdapter:
    """Optional zero-shot Kronos direction confirmer.

    This adapter is deliberately optional. If Kronos dependencies or model weights are
    unavailable, it returns an unavailable result and the caller can continue without it.
    """

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-mini",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k",
        lookback: int = 512,
        pred_len: int = PREDICT_HORIZON_MINUTES,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.lookback = lookback
        self.pred_len = pred_len
        self.temperature = temperature
        self.top_p = top_p
        self.sample_count = sample_count
        self.predictor: Any | None = None
        self.load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        if self.predictor is not None:
            return True
        if self.load_error is not None:
            return False
        try:
            self.predictor = _load_kronos_components(self.model_name, self.tokenizer_name)
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def forecast_direction(self, df: pd.DataFrame) -> KronosForecastResult:
        if not self._ensure_loaded():
            return KronosForecastResult(False, "no_trade", 0.0, None, None, f"kronos_unavailable:{self.load_error}")

        required = ["open", "high", "low", "close"]
        if "timestamp" not in df.columns:
            return KronosForecastResult(False, "no_trade", 0.0, None, None, "missing_timestamp")
        for col in required:
            if col not in df.columns:
                return KronosForecastResult(False, "no_trade", 0.0, None, None, f"missing_{col}")

        data = df.tail(self.lookback).copy()
        if len(data) < min(64, self.pred_len + 10):
            return KronosForecastResult(False, "no_trade", 0.0, None, None, "not_enough_kline_context")

        try:
            x_df = data[[c for c in ["open", "high", "low", "close", "volume", "amount"] if c in data.columns]].copy()
            x_timestamp = pd.to_datetime(data["timestamp"], unit="ms")
            last_ts = x_timestamp.iloc[-1]
            y_timestamp = pd.Series(pd.date_range(last_ts + pd.Timedelta(minutes=1), periods=self.pred_len, freq="1min"))

            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=self.pred_len,
                T=self.temperature,
                top_p=self.top_p,
                sample_count=self.sample_count,
            )

            current_close = float(data.iloc[-1]["close"])
            forecast_close = float(pred_df.iloc[-1]["close"])
            direction = "up" if forecast_close > current_close else "down"
            denominator = max(abs(current_close), 1e-12)
            confidence = min(abs(forecast_close - current_close) / denominator * 100.0, 1.0)
            return KronosForecastResult(True, direction, float(confidence), forecast_close, current_close, "kronos_forecast_ok")
        except Exception as exc:
            return KronosForecastResult(False, "no_trade", 0.0, None, None, f"kronos_predict_error:{type(exc).__name__}:{exc}")
