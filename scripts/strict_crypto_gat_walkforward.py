from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch is optional for tree-based live models.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, PREDICT_HORIZON_MINUTES, RANDOM_STATE  # noqa: E402
from data_download import load_history_csv  # noqa: E402

try:
    from core.hour_rule_pool import wilson_interval  # noqa: E402
except Exception:
    def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
        if total <= 0:
            return None, None
        p = wins / total
        denom = 1.0 + z * z / total
        center = (p + z * z / (2 * total)) / denom
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
        return max(0.0, center - margin), min(1.0, center + margin)


KLINE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    win_rate: float
    signals: int
    signals_per_day: float
    wilson_lower: float


if nn is not None:
    class BtcGraphAttentionClassifier(nn.Module):
        def __init__(self, feature_dim: int, hidden_dim: int, node_mask: np.ndarray, dropout: float) -> None:
            super().__init__()
            self.linear = nn.Linear(feature_dim, hidden_dim)
            self.attention = nn.Linear(hidden_dim * 2, 1)
            self.classifier = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)
            mask = torch.tensor(node_mask.astype(bool), dtype=torch.bool)
            self.register_buffer("node_mask", mask)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.nn.functional.elu(self.linear(x))
            btc = h[:, :1, :].expand(-1, h.shape[1], -1)
            scores = torch.nn.functional.leaky_relu(self.attention(torch.cat([btc, h], dim=-1)).squeeze(-1), negative_slope=0.2)
            scores = scores.masked_fill(~self.node_mask.unsqueeze(0), -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            context = torch.sum(weights * h, dim=1)
            return self.classifier(self.dropout(context)).squeeze(-1)


    class CrossAssetGraphAttentionClassifier(nn.Module):
        def __init__(self, feature_dim: int, hidden_dim: int, adjacency_mask: np.ndarray, dropout: float) -> None:
            super().__init__()
            self.linear = nn.Linear(feature_dim, hidden_dim)
            self.attention = nn.Linear(hidden_dim * 2, 1)
            self.classifier = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)
            mask = torch.tensor(adjacency_mask.astype(bool), dtype=torch.bool)
            self.register_buffer("adjacency_mask", mask)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.nn.functional.elu(self.linear(x))
            target = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
            source = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
            scores = torch.nn.functional.leaky_relu(self.attention(torch.cat([target, source], dim=-1)).squeeze(-1), negative_slope=0.2)
            scores = scores.masked_fill(~self.adjacency_mask.unsqueeze(0), -1e9)
            weights = torch.softmax(scores, dim=2)
            context = torch.matmul(weights, h)
            return self.classifier(self.dropout(context)).squeeze(-1)


def _as_of_to_ms(value: str | None) -> int | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Hong_Kong")
    return int(ts.tz_convert("UTC").timestamp() * 1000)


def _symbol_path(symbol: str) -> Path:
    if symbol.upper() == "BTCUSDT":
        return DATA_DIR / "BTCUSDT_1m_history.csv"
    return DATA_DIR / f"{symbol.upper()}_1m_history.csv"


def _load_symbol(symbol: str) -> pd.DataFrame:
    path = _symbol_path(symbol)
    if symbol.upper() == "BTCUSDT" and path.exists():
        frame = load_history_csv()
    else:
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; rerun with --download-missing")
        frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["timestamp", "close"]).drop_duplicates("timestamp").sort_values("timestamp")


def _download_symbol(symbol: str, days: int, base_url: str) -> None:
    path = _symbol_path(symbol)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty:
        existing["timestamp"] = pd.to_numeric(existing["timestamp"], errors="coerce")
        existing = existing.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    now_ms = int(time.time() * 1000)
    requested_start = now_ms - days * 86_400_000
    rows: list[list] = []
    session = requests.Session()

    def fetch_range(start_ms: int, end_ms: int) -> None:
        while start_ms < end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": "1m",
                "startTime": int(start_ms),
                "endTime": int(end_ms),
                "limit": 1000,
            }
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response = session.get(f"{base_url}/api/v3/klines", params=params, timeout=(5, 15))
                    response.raise_for_status()
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 2:
                        raise
                    time.sleep(0.5 * (attempt + 1))
            if last_error is not None and "response" not in locals():
                raise last_error
            batch = response.json()
            if not batch:
                break
            rows.extend(batch)
            next_start = int(batch[-1][0]) + 60_000
            if next_start <= start_ms:
                raise RuntimeError(f"{symbol} download pagination did not advance")
            start_ms = next_start
            time.sleep(0.03)

    if existing.empty:
        fetch_range(requested_start, now_ms)
    else:
        existing_start = int(existing["timestamp"].min())
        existing_end = int(existing["timestamp"].max())
        if existing_start > requested_start:
            fetch_range(requested_start, existing_start - 60_000)
        if existing_end + 60_000 < now_ms:
            fetch_range(existing_end + 60_000, now_ms)
    if rows:
        fresh = pd.DataFrame(rows, columns=KLINE_COLUMNS).drop(columns=["ignore"])
        combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    else:
        combined = existing
    if combined.empty:
        raise RuntimeError(f"no kline data downloaded for {symbol}")
    combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    combined = combined[combined["timestamp"] >= requested_start].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    print(
        f"[download] {symbol} rows={len(combined)} "
        f"start={pd.to_datetime(combined['timestamp'].min(), unit='ms', utc=True)} "
        f"end={pd.to_datetime(combined['timestamp'].max(), unit='ms', utc=True)}",
        flush=True,
    )


def _resample_bars(frame: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    data = frame.copy()
    data["dt"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
    data = data.set_index("dt")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "quote_asset_volume" in data.columns:
        agg["quote_asset_volume"] = "sum"
    if "taker_buy_base_volume" in data.columns:
        agg["taker_buy_base_volume"] = "sum"
    bars = data.resample(f"{bar_minutes}min", label="right", closed="right").agg(agg).dropna(subset=["open", "high", "low", "close"])
    raw_timestamp = bars.index.astype("int64")
    if len(raw_timestamp) and int(raw_timestamp.max()) > 10_000_000_000_000:
        raw_timestamp = raw_timestamp // 1_000_000
    bars["timestamp"] = raw_timestamp.astype("int64")
    return bars.reset_index(drop=True)


def _node_features(bars: pd.DataFrame, symbol: str, lags: list[int], feature_set: str, lookback_bars: int) -> pd.DataFrame:
    out = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    ret = out["close"].pct_change()
    columns = {"timestamp": out["timestamp"]}
    for lag in lags:
        columns[f"{symbol}_ret_{lag}"] = out["close"].pct_change(lag)
    columns[f"{symbol}_range"] = (out["high"] - out["low"]) / out["close"]
    columns[f"{symbol}_body"] = (out["close"] - out["open"]) / out["open"]
    vol_mean = out["volume"].rolling(20, min_periods=10).mean()
    columns[f"{symbol}_vol_ratio"] = out["volume"] / vol_mean - 1.0
    columns[f"{symbol}_volatility_10"] = ret.rolling(10, min_periods=5).std()
    if "taker_buy_base_volume" in bars.columns:
        taker_buy_ratio = bars["taker_buy_base_volume"] / out["volume"].replace(0, np.nan)
        columns[f"{symbol}_taker_buy_ratio"] = taker_buy_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    if feature_set == "fgat":
        prev_close = out["close"].shift(1).replace(0, np.nan)
        volume_ma5 = out["volume"].rolling(5, min_periods=3).mean().replace(0, np.nan)
        columns[f"{symbol}_norm_open"] = out["open"] / prev_close - 1.0
        columns[f"{symbol}_norm_high"] = out["high"] / prev_close - 1.0
        columns[f"{symbol}_norm_low"] = out["low"] / prev_close - 1.0
        columns[f"{symbol}_norm_close"] = out["close"] / prev_close - 1.0
        columns[f"{symbol}_norm_volume"] = out["volume"] / volume_ma5 - 1.0
        full_range = (out["high"] - out["low"]).replace(0, np.nan)
        columns[f"{symbol}_upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / full_range
        columns[f"{symbol}_lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / full_range
        columns[f"{symbol}_close_pos"] = (out["close"] - out["low"]) / full_range
        windows = sorted({3, 5, 10, 20, max(int(lookback_bars), 2)})
        for window in windows:
            min_periods = max(2, min(window, max(3, window // 2)))
            columns[f"{symbol}_ret_mean_{window}"] = ret.rolling(window, min_periods=min_periods).mean()
            columns[f"{symbol}_ret_std_{window}"] = ret.rolling(window, min_periods=min_periods).std()
            columns[f"{symbol}_ret_min_{window}"] = ret.rolling(window, min_periods=min_periods).min()
            columns[f"{symbol}_ret_max_{window}"] = ret.rolling(window, min_periods=min_periods).max()
            columns[f"{symbol}_momentum_{window}"] = out["close"] / out["close"].shift(window).replace(0, np.nan) - 1.0
            columns[f"{symbol}_volume_ratio_{window}"] = out["volume"] / out["volume"].rolling(window, min_periods=min_periods).mean().replace(0, np.nan) - 1.0
    return pd.DataFrame(columns)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / max(temperature, 1e-6)
    scaled = scaled - np.nanmax(scaled)
    weights = np.exp(scaled)
    total = np.nansum(weights)
    if not np.isfinite(total) or total <= 0:
        return np.full(len(values), 1.0 / max(len(values), 1))
    return weights / total


def _build_graph_features(
    feature_frame: pd.DataFrame,
    returns_frame: pd.DataFrame,
    *,
    symbols: list[str],
    train_start: int,
    graph_end: int,
    top_k: int,
    attention_temperature: float,
) -> tuple[pd.DataFrame, dict]:
    train_returns = returns_frame[(returns_frame["timestamp"] >= train_start) & (returns_frame["timestamp"] < graph_end)].copy()
    return_cols = [f"{symbol}_target_ret" for symbol in symbols]
    corr = train_returns[return_cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    btc_col = "BTCUSDT_target_ret"
    if btc_col not in corr:
        raise ValueError("BTCUSDT must be included in --symbols")
    neighbors = [col.replace("_target_ret", "") for col in corr[btc_col].abs().sort_values(ascending=False).index if col != btc_col]
    neighbors = neighbors[: max(top_k, 1)]
    raw_scores = np.array([abs(float(corr.loc[btc_col, f"{symbol}_target_ret"])) for symbol in neighbors], dtype=float)
    weights = _softmax(raw_scores, attention_temperature)
    graph_columns: dict[str, object] = {"timestamp": feature_frame["timestamp"].to_numpy(dtype=np.int64)}
    btc_feature_cols = [
        col
        for col in feature_frame.columns
        if col.startswith("BTCUSDT_") and not col.endswith("_target_ret") and not col.endswith("_future_return")
    ]
    for column in btc_feature_cols:
        graph_columns[column] = feature_frame[column].to_numpy(dtype=float)
    aggregate_feature_cols = [col for col in btc_feature_cols if "_xsec_" not in col]
    suffixes = sorted({col.split("_", 1)[1] for col in aggregate_feature_cols})
    for suffix in suffixes:
        aggregate = np.zeros(len(feature_frame), dtype=float)
        available_weight = np.zeros(len(feature_frame), dtype=float)
        for symbol, weight in zip(neighbors, weights):
            column = f"{symbol}_{suffix}"
            if column not in feature_frame.columns:
                continue
            values = pd.to_numeric(feature_frame[column], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values)
            aggregate[mask] += values[mask] * float(weight)
            available_weight[mask] += float(weight)
        aggregate = np.divide(aggregate, available_weight, out=np.full(len(feature_frame), np.nan), where=available_weight > 0)
        graph_columns[f"gat_neighbor_{suffix}"] = aggregate
        btc_column = f"BTCUSDT_{suffix}"
        if btc_column in feature_frame.columns:
            graph_columns[f"gat_spread_{suffix}"] = feature_frame[btc_column].to_numpy(dtype=float) - aggregate
    graph_columns["gat_abs_corr_mean"] = np.full(len(feature_frame), float(np.nanmean(raw_scores)) if len(raw_scores) else 0.0)
    graph_columns["gat_abs_corr_max"] = np.full(len(feature_frame), float(np.nanmax(raw_scores)) if len(raw_scores) else 0.0)
    graph = pd.DataFrame(graph_columns)
    metadata = {
        "neighbors": neighbors,
        "attention_weights": {symbol: float(weight) for symbol, weight in zip(neighbors, weights)},
        "btc_abs_corr": {symbol: float(score) for symbol, score in zip(neighbors, raw_scores)},
    }
    return graph, metadata


def _prepare_dataset(
    symbols: list[str],
    bar_minutes: int,
    horizon_minutes: int,
    as_of_ms: int | None,
    *,
    feature_set: str = "compact",
    lookback_bars: int = 30,
    xsec_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if horizon_minutes <= 0:
        raise ValueError(f"horizon_minutes must be positive: {horizon_minutes}")
    if bar_minutes <= 0:
        raise ValueError(f"bar_minutes must be positive: {bar_minutes}")
    horizon_bars = max(1, int(math.ceil(horizon_minutes / bar_minutes)))
    feature_parts = []
    return_parts = []
    lags = [1, 2, 4, 8, 16]
    for symbol in symbols:
        bars = _resample_bars(_load_symbol(symbol), bar_minutes)
        if as_of_ms is not None:
            bars = bars[bars["timestamp"] <= as_of_ms].copy()
        features = _node_features(bars, symbol, lags, feature_set, lookback_bars)
        labels = bars[["timestamp", "close"]].copy()
        labels[f"{symbol}_future_return"] = labels["close"].shift(-horizon_bars) / labels["close"] - 1.0
        features = pd.merge(features, labels[["timestamp", f"{symbol}_future_return"]], on="timestamp", how="inner")
        returns = bars[["timestamp", "close"]].copy()
        returns[f"{symbol}_target_ret"] = returns["close"].pct_change()
        feature_parts.append(features)
        return_parts.append(returns[["timestamp", f"{symbol}_target_ret"]])
    merged = feature_parts[0]
    returns = return_parts[0]
    for part in feature_parts[1:]:
        merged = pd.merge(merged, part, on="timestamp", how="inner")
    for part in return_parts[1:]:
        returns = pd.merge(returns, part, on="timestamp", how="inner")
    if xsec_features:
        relative_suffixes = ["ret_1", "ret_2", "ret_4", "ret_8", "ret_16", "range", "body", "vol_ratio", "volatility_10"]
        if feature_set == "fgat":
            relative_suffixes.extend(
                [
                    "norm_open",
                    "norm_high",
                    "norm_low",
                    "norm_close",
                    "norm_volume",
                    "close_pos",
                    f"momentum_{max(int(lookback_bars), 2)}",
                    f"volume_ratio_{max(int(lookback_bars), 2)}",
                ]
            )
        relative_columns: dict[str, object] = {}
        for suffix in relative_suffixes:
            symbol_columns = [f"{symbol}_{suffix}" for symbol in symbols if f"{symbol}_{suffix}" in merged.columns]
            btc_column = f"BTCUSDT_{suffix}"
            if btc_column not in merged.columns or len(symbol_columns) < 3:
                continue
            values = merged[symbol_columns]
            mean = values.mean(axis=1)
            std = values.std(axis=1).replace(0, np.nan)
            rank_pct = values.rank(axis=1, pct=True)[btc_column]
            relative_columns[f"BTCUSDT_xsec_mean_{suffix}"] = mean
            relative_columns[f"BTCUSDT_xsec_z_{suffix}"] = (merged[btc_column] - mean) / std
            relative_columns[f"BTCUSDT_xsec_rank_{suffix}"] = rank_pct
        if relative_columns:
            merged = pd.concat([merged, pd.DataFrame(relative_columns)], axis=1)
    btc_close = _resample_bars(_load_symbol("BTCUSDT"), bar_minutes)[["timestamp", "close"]]
    if as_of_ms is not None:
        btc_close = btc_close[btc_close["timestamp"] <= as_of_ms].copy()
    btc_close["future_close"] = btc_close["close"].shift(-horizon_bars)
    btc_close["future_return"] = btc_close["future_close"] / btc_close["close"] - 1.0
    merged = pd.merge(merged, btc_close[["timestamp", "future_return"]], on="timestamp", how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).sort_values("timestamp").reset_index(drop=True)
    returns = returns.replace([np.inf, -np.inf], np.nan).sort_values("timestamp").reset_index(drop=True)
    return merged, returns


def _model(name: str):
    if name == "lgb":
        return LGBMClassifier(
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=12,
            max_depth=4,
            min_child_samples=60,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=3.0,
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbose=-1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=260,
            max_depth=7,
            min_samples_leaf=40,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_alpha=0.5,
            reg_lambda=4.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        )
    raise ValueError(f"unknown model: {name}")


def _fit_predict_tree_probability(model_name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y_train = (train["future_return"] > 0).astype(int)
    if y_train.nunique() < 2:
        raise ValueError("single_class_train")
    if model_name == "extra_xgb_avg":
        extra = _model("extra_trees")
        xgb = _model("xgboost")
        extra.fit(train[feature_cols], y_train)
        xgb.fit(train[feature_cols], y_train)
        val_prob = 0.5 * extra.predict_proba(val[feature_cols])[:, 1] + 0.5 * xgb.predict_proba(val[feature_cols])[:, 1]
        test_prob = 0.5 * extra.predict_proba(test[feature_cols])[:, 1] + 0.5 * xgb.predict_proba(test[feature_cols])[:, 1]
        return val_prob, test_prob
    model = _model(model_name)
    model.fit(train[feature_cols], y_train)
    return model.predict_proba(val[feature_cols])[:, 1], model.predict_proba(test[feature_cols])[:, 1]


def _common_node_suffixes(frame: pd.DataFrame, symbols: list[str]) -> list[str]:
    suffix_sets = []
    for symbol in symbols:
        prefix = f"{symbol}_"
        suffix_sets.append({column[len(prefix) :] for column in frame.columns if column.startswith(prefix)})
    common = set.intersection(*suffix_sets) if suffix_sets else set()
    return sorted(suffix for suffix in common if not suffix.endswith("target_ret") and not suffix.endswith("future_return"))


def _node_tensor(frame: pd.DataFrame, symbols: list[str], suffixes: list[str]) -> np.ndarray:
    arrays = []
    for symbol in symbols:
        columns = [f"{symbol}_{suffix}" for suffix in suffixes]
        arrays.append(frame[columns].to_numpy(dtype=np.float32))
    return np.stack(arrays, axis=1)


def _adjacency_mask(returns_frame: pd.DataFrame, symbols: list[str], train_start: int, graph_end: int, top_k: int) -> np.ndarray:
    train_returns = returns_frame[(returns_frame["timestamp"] >= train_start) & (returns_frame["timestamp"] < graph_end)].copy()
    return_cols = [f"{symbol}_target_ret" for symbol in symbols]
    corr = train_returns[return_cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mask = np.eye(len(symbols), dtype=bool)
    for i, symbol in enumerate(symbols):
        col = f"{symbol}_target_ret"
        ranked = [name for name in corr[col].abs().sort_values(ascending=False).index if name != col]
        for neighbor_col in ranked[: max(top_k, 1)]:
            neighbor = neighbor_col.replace("_target_ret", "")
            if neighbor in symbols:
                mask[i, symbols.index(neighbor)] = True
    return mask


def _torch_gat_probabilities(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    symbols: list[str],
    neighbors: list[str],
    epochs: int,
    hidden_size: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    dropout: float,
    device_name: str,
) -> np.ndarray:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("torch is required for --model torch_gat")
    suffixes = _common_node_suffixes(train, symbols)
    node_columns = [f"{symbol}_{suffix}" for symbol in symbols for suffix in suffixes]
    train = train.dropna(subset=node_columns + ["future_return"]).copy()
    predict = predict.dropna(subset=node_columns + ["future_return"]).copy()
    if len(train) < 300 or predict.empty or not suffixes:
        return np.array([], dtype=float)

    x_train = _node_tensor(train, symbols, suffixes)
    x_pred = _node_tensor(predict, symbols, suffixes)
    y_train = (train["future_return"].to_numpy(dtype=float) > 0).astype(np.float32)
    if len(np.unique(y_train)) < 2:
        return np.array([], dtype=float)

    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    x_train = np.nan_to_num((x_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x_pred = np.nan_to_num((x_pred - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    node_mask = np.zeros(len(symbols), dtype=bool)
    node_mask[0] = True
    for neighbor in neighbors:
        if neighbor in symbols:
            node_mask[symbols.index(neighbor)] = True

    device = torch.device(device_name)
    torch.manual_seed(RANDOM_STATE)
    model = BtcGraphAttentionClassifier(len(suffixes), hidden_size, node_mask, dropout).to(device)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x_pred), batch_size):
            batch = torch.from_numpy(x_pred[start : start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([], dtype=float)


def _torch_cross_asset_gat_probabilities(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    returns_frame: pd.DataFrame,
    symbols: list[str],
    train_start: int,
    graph_end: int,
    top_k: int,
    epochs: int,
    hidden_size: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    dropout: float,
    device_name: str,
) -> np.ndarray:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("torch is required for --model torch_cross_gat")
    suffixes = _common_node_suffixes(train, symbols)
    node_columns = [f"{symbol}_{suffix}" for symbol in symbols for suffix in suffixes]
    label_columns = [f"{symbol}_future_return" for symbol in symbols]
    train = train.dropna(subset=node_columns + label_columns).copy()
    predict = predict.dropna(subset=node_columns).copy()
    if len(train) < 300 or predict.empty or not suffixes:
        return np.array([], dtype=float)

    x_train = _node_tensor(train, symbols, suffixes)
    x_pred = _node_tensor(predict, symbols, suffixes)
    y_train = (train[label_columns].to_numpy(dtype=float) > 0).astype(np.float32)
    if np.unique(y_train).size < 2:
        return np.array([], dtype=float)

    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    x_train = np.nan_to_num((x_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x_pred = np.nan_to_num((x_pred - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    adjacency = _adjacency_mask(returns_frame, symbols, train_start, graph_end, top_k)
    device = torch.device(device_name)
    torch.manual_seed(RANDOM_STATE)
    model = CrossAssetGraphAttentionClassifier(len(suffixes), hidden_size, adjacency, dropout).to(device)
    positives = float(y_train.sum())
    negatives = float(y_train.size - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x_pred), batch_size):
            batch = torch.from_numpy(x_pred[start : start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch))[:, 0].detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([], dtype=float)


def _eval_threshold(prob: np.ndarray, future_return: np.ndarray, threshold: float, direction_filter: str = "both") -> tuple[int, int, float]:
    up = prob >= threshold
    down = prob <= 1.0 - threshold
    if direction_filter == "up":
        down = np.zeros_like(down, dtype=bool)
    elif direction_filter == "down":
        up = np.zeros_like(up, dtype=bool)
    active = up | down
    if not active.any():
        return 0, 0, math.nan
    correct = np.where(up, future_return > 0, future_return < 0)
    signals = int(active.sum())
    wins = int(correct[active].sum())
    return signals, wins, wins / signals


def _select_threshold(
    prob: np.ndarray,
    future_return: np.ndarray,
    *,
    validation_days: float,
    min_win_rate: float,
    min_signals_per_day: float,
    direction_filter: str = "both",
) -> ThresholdSelection:
    best: ThresholdSelection | None = None
    for threshold in [round(value, 2) for value in np.arange(0.52, 0.951, 0.01)]:
        signals, wins, win_rate = _eval_threshold(prob, future_return, threshold, direction_filter)
        if not signals or not np.isfinite(win_rate):
            continue
        signals_per_day = signals / max(validation_days, 1e-9)
        if win_rate < min_win_rate or signals_per_day < min_signals_per_day:
            continue
        lower, _ = wilson_interval(wins, signals)
        item = ThresholdSelection(threshold, win_rate, signals, signals_per_day, float(lower or 0.0))
        if best is None or (item.wilson_lower, item.win_rate, item.signals, item.threshold) > (
            best.wilson_lower,
            best.win_rate,
            best.signals,
            best.threshold,
        ):
            best = item
    return best or ThresholdSelection(1.01, math.nan, 0, 0.0, 0.0)


def _signals_frame(prob: np.ndarray, frame: pd.DataFrame, threshold: float, fold: int, direction_filter: str = "both") -> pd.DataFrame:
    up = prob >= threshold
    down = prob <= 1.0 - threshold
    if direction_filter == "up":
        down = np.zeros_like(down, dtype=bool)
    elif direction_filter == "down":
        up = np.zeros_like(up, dtype=bool)
    active = up | down
    if not active.any():
        return pd.DataFrame()
    future_return = frame["future_return"].to_numpy(dtype=float)
    direction = np.where(up, "up", "down")
    correct = np.where(up, future_return > 0, future_return < 0)
    return pd.DataFrame(
        {
            "fold": fold,
            "timestamp": frame.loc[active, "timestamp"].astype("int64").to_numpy(),
            "probability": prob[active],
            "threshold": threshold,
            "direction": direction[active],
            "future_return": future_return[active],
            "correct": correct[active],
        }
    )


def _score(signals: pd.DataFrame, days: float) -> dict:
    if signals.empty:
        return {"signals": 0, "wins": 0, "win_rate": None, "signals_per_day": 0.0}
    wins = int(signals["correct"].astype(bool).sum())
    return {
        "signals": int(len(signals)),
        "wins": wins,
        "win_rate": wins / len(signals),
        "signals_per_day": len(signals) / max(days, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict walk-forward CryptoGAT-style cross-asset graph model for BTC 30m direction.")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--inner-validation-days", type=int, default=7)
    parser.add_argument("--bar-minutes", type=int, default=1)
    parser.add_argument("--horizon-minutes", type=int, default=PREDICT_HORIZON_MINUTES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--attention-temperature", type=float, default=0.2)
    parser.add_argument("--feature-set", choices=["compact", "fgat"], default="compact")
    parser.add_argument("--lookback-bars", type=int, default=30)
    parser.add_argument("--xsec-features", action="store_true")
    parser.add_argument("--model", choices=["lgb", "extra_trees", "xgboost", "extra_xgb_avg", "torch_gat", "torch_cross_gat"], default="lgb")
    parser.add_argument("--gat-epochs", type=int, default=40)
    parser.add_argument("--gat-hidden-size", type=int, default=64)
    parser.add_argument("--gat-learning-rate", type=float, default=0.00025)
    parser.add_argument("--gat-weight-decay", type=float, default=0.0001)
    parser.add_argument("--gat-batch-size", type=int, default=256)
    parser.add_argument("--gat-dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-threshold-win-rate", type=float, default=0.58)
    parser.add_argument("--min-threshold-signals-per-day", type=float, default=0.2)
    parser.add_argument("--direction-filter", choices=["both", "up", "down"], default="both")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--download-days", type=int, default=180)
    parser.add_argument("--binance-base-url", default="https://api.binance.com")
    parser.add_argument("--output", default=str(DATA_DIR / "strict_crypto_gat_summary.json"))
    parser.add_argument("--signals-output", default=str(DATA_DIR / "strict_crypto_gat_signals.csv"))
    args = parser.parse_args()

    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    if "BTCUSDT" not in symbols:
        symbols.insert(0, "BTCUSDT")
    if args.download_missing:
        for symbol in symbols:
            if symbol == "BTCUSDT" and _symbol_path(symbol).exists():
                continue
            _download_symbol(symbol, args.download_days, args.binance_base_url)

    as_of_ms = _as_of_to_ms(args.as_of)
    data, returns = _prepare_dataset(
        symbols,
        args.bar_minutes,
        args.horizon_minutes,
        as_of_ms,
        feature_set=args.feature_set,
        lookback_bars=args.lookback_bars,
        xsec_features=args.xsec_features,
    )
    data = data.dropna(subset=["future_return"]).reset_index(drop=True)
    latest = int(data["timestamp"].max())
    if as_of_ms is not None:
        latest = min(latest, as_of_ms - args.horizon_minutes * 60_000)
    test_start = latest - args.validation_days * 86_400_000
    fold_ms = args.validation_days * 86_400_000 // args.folds
    signal_frames = []
    fold_rows = []
    graph_rows = []

    for fold in range(1, args.folds + 1):
        fold_start = test_start + (fold - 1) * fold_ms
        fold_end = test_start + (fold * fold_ms if fold < args.folds else args.validation_days * 86_400_000)
        inner_val_start = fold_start - args.inner_validation_days * 86_400_000
        train_start = inner_val_start - args.train_days * 86_400_000
        graph_features, graph_meta = _build_graph_features(
            data,
            returns,
            symbols=symbols,
            train_start=train_start,
            graph_end=inner_val_start,
            top_k=args.top_k,
            attention_temperature=args.attention_temperature,
        )
        if args.model in {"torch_gat", "torch_cross_gat"}:
            suffixes = _common_node_suffixes(data, symbols)
            feature_cols = [f"{symbol}_{suffix}" for symbol in symbols for suffix in suffixes]
            label_cols = [f"{symbol}_future_return" for symbol in symbols if f"{symbol}_future_return" in data.columns]
            merged = data[["timestamp", "future_return"] + feature_cols + label_cols].copy()
            train = merged[(merged["timestamp"] >= train_start) & (merged["timestamp"] < inner_val_start)].copy()
            val = merged[(merged["timestamp"] >= inner_val_start) & (merged["timestamp"] < fold_start)].copy()
            test = merged[(merged["timestamp"] >= fold_start) & (merged["timestamp"] < fold_end)].copy()
            for frame in (train, val, test):
                frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
                frame.dropna(subset=feature_cols + ["future_return"], inplace=True)
            if len(train) < 300 or len(val) < 30 or test.empty:
                fold_rows.append({"fold": fold, "skipped": True, "train_rows": len(train), "val_rows": len(val), "test_rows": len(test)})
                continue
            combined_predict = pd.concat([val, test], ignore_index=True)
            if args.model == "torch_cross_gat":
                combined_prob = _torch_cross_asset_gat_probabilities(
                    train,
                    combined_predict,
                    returns_frame=returns,
                    symbols=symbols,
                    train_start=train_start,
                    graph_end=inner_val_start,
                    top_k=args.top_k,
                    epochs=args.gat_epochs,
                    hidden_size=args.gat_hidden_size,
                    learning_rate=args.gat_learning_rate,
                    weight_decay=args.gat_weight_decay,
                    batch_size=args.gat_batch_size,
                    dropout=args.gat_dropout,
                    device_name=args.device,
                )
            else:
                combined_prob = _torch_gat_probabilities(
                    train,
                    combined_predict,
                    symbols=symbols,
                    neighbors=graph_meta["neighbors"],
                    epochs=args.gat_epochs,
                    hidden_size=args.gat_hidden_size,
                    learning_rate=args.gat_learning_rate,
                    weight_decay=args.gat_weight_decay,
                    batch_size=args.gat_batch_size,
                    dropout=args.gat_dropout,
                    device_name=args.device,
                )
            if len(combined_prob) != len(combined_predict):
                fold_rows.append({"fold": fold, "skipped": True, "reason": "torch_probability_mismatch"})
                continue
            val_prob = combined_prob[: len(val)]
            test_prob_for_torch = combined_prob[len(val) :]
        else:
            merged = pd.merge(data[["timestamp", "future_return"]], graph_features, on="timestamp", how="inner")
            feature_cols = [col for col in merged.columns if col not in {"timestamp", "future_return"}]
            train = merged[(merged["timestamp"] >= train_start) & (merged["timestamp"] < inner_val_start)].copy()
            val = merged[(merged["timestamp"] >= inner_val_start) & (merged["timestamp"] < fold_start)].copy()
            test = merged[(merged["timestamp"] >= fold_start) & (merged["timestamp"] < fold_end)].copy()
            for frame in (train, val, test):
                frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
                frame.dropna(subset=feature_cols + ["future_return"], inplace=True)
            if len(train) < 300 or len(val) < 30 or test.empty:
                fold_rows.append({"fold": fold, "skipped": True, "train_rows": len(train), "val_rows": len(val), "test_rows": len(test)})
                continue
            try:
                val_prob, test_prob_for_tree = _fit_predict_tree_probability(args.model, train, val, test, feature_cols)
            except ValueError as exc:
                fold_rows.append({"fold": fold, "skipped": True, "reason": str(exc)})
                continue
        selected = _select_threshold(
            val_prob,
            val["future_return"].to_numpy(dtype=float),
            validation_days=args.inner_validation_days,
            min_win_rate=args.min_threshold_win_rate,
            min_signals_per_day=args.min_threshold_signals_per_day,
            direction_filter=args.direction_filter,
        )
        if args.model in {"torch_gat", "torch_cross_gat"}:
            test_prob = test_prob_for_torch
        else:
            test_prob = test_prob_for_tree
        signals = _signals_frame(test_prob, test, selected.threshold, fold, args.direction_filter)
        if not signals.empty:
            signal_frames.append(signals)
        test_score = _score(signals, (fold_end - fold_start) / 86_400_000)
        fold_rows.append(
            {
                "fold": fold,
                "skipped": False,
                "threshold": selected.threshold,
                "threshold_val_win_rate": selected.win_rate,
                "threshold_val_signals": selected.signals,
                "threshold_val_wilson_lower": selected.wilson_lower,
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                **{f"test_{key}": value for key, value in test_score.items()},
            }
        )
        graph_rows.append({"fold": fold, **graph_meta})
        print(
            f"[crypto_gat] fold={fold}/{args.folds} threshold={selected.threshold:.2f} "
            f"val={selected.win_rate if np.isfinite(selected.win_rate) else np.nan:.3f}/{selected.signals} "
            f"test={test_score['wins']}/{test_score['signals']} neighbors={graph_meta['neighbors']}",
            flush=True,
        )

    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    if not signals.empty:
        local = pd.to_datetime(signals["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Hong_Kong")
        by_hour = [
            {"hour": int(hour), **_score(group, args.validation_days)}
            for hour, group in signals.assign(hour=local.dt.hour).groupby("hour", sort=True)
        ]
        by_direction = [
            {"direction": str(direction), **_score(group, args.validation_days)}
            for direction, group in signals.groupby("direction", sort=True)
        ]
    else:
        by_hour = []
        by_direction = []
    payload = {
        "parameters": vars(args),
        "symbols": symbols,
        "range": {
            "data_start": pd.to_datetime(int(data["timestamp"].min()), unit="ms", utc=True).tz_convert("Asia/Hong_Kong").isoformat(),
            "data_latest": pd.to_datetime(latest, unit="ms", utc=True).tz_convert("Asia/Hong_Kong").isoformat(),
            "test_start": pd.to_datetime(test_start, unit="ms", utc=True).tz_convert("Asia/Hong_Kong").isoformat(),
        },
        "overall": _score(signals, args.validation_days),
        "folds": fold_rows,
        "graphs": graph_rows,
        "by_hour": by_hour,
        "by_direction": by_direction,
        "strictness": {
            "graph_edges_use_only_training_window_returns": True,
            "threshold_selected_only_from_past_inner_validation": True,
            "test_period_never_used_for_training_threshold_or_graph": True,
            "target": f"BTCUSDT direction after {args.horizon_minutes} minutes using {args.bar_minutes}m bars",
            "bar_minutes": args.bar_minutes,
            "horizon_minutes": args.horizon_minutes,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not signals.empty:
        signals.to_csv(args.signals_output, index=False)
    else:
        pd.DataFrame().to_csv(args.signals_output, index=False)
    print(json.dumps({"overall": payload["overall"], "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
