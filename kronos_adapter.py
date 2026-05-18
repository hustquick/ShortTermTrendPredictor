# kronos_adapter.py

from dataclasses import dataclass
from contextlib import contextmanager
import multiprocessing as mp
from pathlib import Path
import signal
import sys
from typing import Any

import pandas as pd

from config import (
    KRONOS_DEVICE,
    KRONOS_LOCAL_FILES_ONLY,
    KRONOS_LOOKBACK,
    KRONOS_LOAD_TIMEOUT_SECONDS,
    KRONOS_MAX_CONTEXT,
    KRONOS_MODEL_NAME,
    KRONOS_PREDICT_TIMEOUT_SECONDS,
    KRONOS_TOKENIZER_NAME,
    KRONOS_USE_SUBPROCESS,
    PREDICT_HORIZON_MINUTES,
)


@dataclass
class KronosForecastResult:
    available: bool
    direction: str
    confidence: float
    forecast_close: float | None
    current_close: float | None
    reason: str


class KronosTimeoutError(TimeoutError):
    pass


@contextmanager
def _time_limit(seconds: int | None, label: str):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise KronosTimeoutError(f"{label}_timeout_{seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _ensure_kronos_source_path():
    candidates = [
        Path(__file__).resolve().parent.parent / "Kronos",
        Path("/Users/zhangcheng/btc/Kronos"),
    ]
    for path in candidates:
        if (path / "model" / "kronos.py").exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return path
    return None


def _load_kronos_components(
    model_name: str,
    tokenizer_name: str,
    local_files_only: bool,
    device: str | None,
    max_context: int,
):
    _ensure_kronos_source_path()
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    except Exception as exc:
        raise ImportError(
            "Kronos is not installed or not importable. "
            "Clone/install https://github.com/shiyu-coder/Kronos under /Users/zhangcheng/btc/Kronos "
            "or add it to PYTHONPATH."
        ) from exc

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
    model = Kronos.from_pretrained(model_name, local_files_only=local_files_only)
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)


def _forecast_worker(payload: dict, queue):
    try:
        adapter = KronosAdapter(
            model_name=payload["model_name"],
            tokenizer_name=payload["tokenizer_name"],
            lookback=payload["lookback"],
            pred_len=payload["pred_len"],
            temperature=payload["temperature"],
            top_p=payload["top_p"],
            sample_count=payload["sample_count"],
            local_files_only=payload["local_files_only"],
            device=payload["device"],
            max_context=payload["max_context"],
            load_timeout_seconds=payload["load_timeout_seconds"],
            predict_timeout_seconds=payload["predict_timeout_seconds"],
            use_subprocess=False,
        )
        result = adapter.forecast_direction(payload["df"])
        queue.put(result)
    except Exception as exc:
        queue.put(
            KronosForecastResult(
                False,
                "no_trade",
                0.0,
                None,
                None,
                f"kronos_worker_error:{type(exc).__name__}:{exc}",
            )
        )


class KronosAdapter:
    """Optional zero-shot Kronos direction confirmer.

    This adapter is deliberately optional. If Kronos dependencies or model weights are
    unavailable, it returns an unavailable result and the caller can continue without it.
    """

    def __init__(
        self,
        model_name: str = KRONOS_MODEL_NAME,
        tokenizer_name: str = KRONOS_TOKENIZER_NAME,
        lookback: int = KRONOS_LOOKBACK,
        pred_len: int = PREDICT_HORIZON_MINUTES,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        local_files_only: bool = KRONOS_LOCAL_FILES_ONLY,
        device: str | None = KRONOS_DEVICE,
        max_context: int = KRONOS_MAX_CONTEXT,
        load_timeout_seconds: int = KRONOS_LOAD_TIMEOUT_SECONDS,
        predict_timeout_seconds: int = KRONOS_PREDICT_TIMEOUT_SECONDS,
        use_subprocess: bool = KRONOS_USE_SUBPROCESS,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.lookback = lookback
        self.pred_len = pred_len
        self.temperature = temperature
        self.top_p = top_p
        self.sample_count = sample_count
        self.local_files_only = local_files_only
        self.device = device
        self.max_context = max_context
        self.load_timeout_seconds = load_timeout_seconds
        self.predict_timeout_seconds = predict_timeout_seconds
        self.use_subprocess = use_subprocess
        self.predictor: Any | None = None
        self.load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        if self.predictor is not None:
            return True
        if self.load_error is not None:
            return False
        try:
            with _time_limit(self.load_timeout_seconds, "kronos_load"):
                self.predictor = _load_kronos_components(
                    self.model_name,
                    self.tokenizer_name,
                    self.local_files_only,
                    self.device,
                    self.max_context,
                )
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def forecast_direction(self, df: pd.DataFrame) -> KronosForecastResult:
        if self.use_subprocess:
            return self._forecast_direction_subprocess(df)

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
            x_df = data[[c for c in ["open", "high", "low", "close", "volume"] if c in data.columns]].copy()
            if "amount" in data.columns:
                x_df["amount"] = data["amount"]
            elif "quote_asset_volume" in data.columns:
                x_df["amount"] = data["quote_asset_volume"]
            x_timestamp = pd.to_datetime(data["timestamp"], unit="ms")
            last_ts = x_timestamp.iloc[-1]
            y_timestamp = pd.Series(pd.date_range(last_ts + pd.Timedelta(minutes=1), periods=self.pred_len, freq="1min"))

            with _time_limit(self.predict_timeout_seconds, "kronos_predict"):
                pred_df = self.predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=self.pred_len,
                    T=self.temperature,
                    top_p=self.top_p,
                    sample_count=self.sample_count,
                    verbose=False,
                )

            current_close = float(data.iloc[-1]["close"])
            forecast_close = float(pred_df.iloc[-1]["close"])
            direction = "up" if forecast_close > current_close else "down"
            denominator = max(abs(current_close), 1e-12)
            confidence = min(abs(forecast_close - current_close) / denominator * 100.0, 1.0)
            return KronosForecastResult(True, direction, float(confidence), forecast_close, current_close, "kronos_forecast_ok")
        except Exception as exc:
            return KronosForecastResult(False, "no_trade", 0.0, None, None, f"kronos_predict_error:{type(exc).__name__}:{exc}")

    def _forecast_direction_subprocess(self, df: pd.DataFrame) -> KronosForecastResult:
        timeout = max(1, int(self.load_timeout_seconds) + int(self.predict_timeout_seconds))
        context = mp.get_context("fork")
        queue = context.Queue(maxsize=1)
        payload = {
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "lookback": self.lookback,
            "pred_len": self.pred_len,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "sample_count": self.sample_count,
            "local_files_only": self.local_files_only,
            "device": self.device,
            "max_context": self.max_context,
            "load_timeout_seconds": self.load_timeout_seconds,
            "predict_timeout_seconds": self.predict_timeout_seconds,
            "df": df.tail(max(self.lookback, self.pred_len + 64)).copy(),
        }
        process = context.Process(target=_forecast_worker, args=(payload, queue), daemon=True)
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            return KronosForecastResult(
                False,
                "no_trade",
                0.0,
                None,
                None,
                f"kronos_subprocess_timeout_{timeout}s",
            )

        if not queue.empty():
            return queue.get()

        return KronosForecastResult(
            False,
            "no_trade",
            0.0,
            None,
            None,
            f"kronos_subprocess_exitcode_{process.exitcode}",
        )
