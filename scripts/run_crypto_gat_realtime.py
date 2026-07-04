from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, PREDICT_HORIZON_MINUTES, WECHAT_REQUEST_TIMEOUT, WECHAT_WEBHOOK_URL  # noqa: E402
from scripts.strict_crypto_gat_walkforward import (  # noqa: E402
    CrossAssetGraphAttentionClassifier,
    DataLoader,
    KLINE_COLUMNS,
    RANDOM_STATE,
    TensorDataset,
    _build_graph_features,
    _adjacency_mask,
    _download_symbol,
    _eval_threshold,
    _fit_predict_tree_probability,
    _common_node_suffixes,
    _load_symbol,
    _model,
    _node_features,
    _prepare_dataset,
    _resample_bars,
    _score,
    _select_threshold,
    _symbol_path,
    _torch_cross_asset_gat_probabilities,
    torch,
    nn,
)


STRATEGY_NAME = "crypto_gat"
LIVE_CSV = DATA_DIR / "crypto_gat_live_predictions.csv"
LIVE_STATE = DATA_DIR / "crypto_gat_live_state.json"
LIVE_COLUMNS = [
    "prediction_id",
    "strategy",
    "signal_timestamp",
    "signal_time",
    "validation_timestamp",
    "validation_time",
    "signal_price",
    "direction",
    "confidence",
    "up_signal_probability",
    "down_signal_probability",
    "direction_edge",
    "threshold",
    "threshold_val_win_rate",
    "threshold_val_signals",
    "threshold_val_wilson_lower",
    "neighbors",
    "attention_weights",
    "reason",
    "validation_status",
    "actual_direction",
    "validation_price",
    "is_correct",
    "notified",
]


@dataclass
class LiveModelCache:
    model_name: str
    trained_at_wall: float
    trained_at_timestamp: int
    feature_cols: list[str]
    symbols: list[str]
    threshold: float
    threshold_val_win_rate: float
    threshold_val_signals: int
    threshold_val_wilson_lower: float
    neighbors: list[str]
    attention_weights: dict
    reason: str
    model: object | None = None
    extra_model: object | None = None
    xgb_model: object | None = None
    suffixes: list[str] | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    batch_size: int = 256
    device_name: str = "cpu"


def _ms_to_hk(ms: int) -> str:
    return pd.to_datetime(int(ms), unit="ms", utc=True).tz_convert("Asia/Hong_Kong").strftime("%Y-%m-%d %H:%M:%S")


def _cache_path(args: argparse.Namespace) -> Path:
    safe_model = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.model)
    return DATA_DIR / f"crypto_gat_live_model_{safe_model}_{int(args.horizon_minutes)}m.pkl"


def _save_model_cache(cache: LiveModelCache, args: argparse.Namespace) -> None:
    path = _cache_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(cache, handle)
    tmp.replace(path)
    print(f"[crypto_gat_live] saved cached model: {path}", flush=True)


def _load_model_cache(args: argparse.Namespace) -> LiveModelCache | None:
    path = _cache_path(args)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("rb") as handle:
            cache = pickle.load(handle)
    except Exception as exc:
        print(f"[crypto_gat_live] cached model load failed: {type(exc).__name__}: {exc}", flush=True)
        return None
    if not isinstance(cache, LiveModelCache):
        print("[crypto_gat_live] cached model ignored: invalid type", flush=True)
        return None
    if cache.model_name != args.model:
        print("[crypto_gat_live] cached model ignored: parameter mismatch", flush=True)
        return None
    if cache.threshold_val_signals <= 0 or not np.isfinite(cache.threshold_val_win_rate):
        print("[crypto_gat_live] cached model ignored: invalid validation stats", flush=True)
        return None
    if cache.model is not None and hasattr(cache.model, "eval"):
        cache.model.eval()
    print(
        "[crypto_gat_live] loaded cached model "
        f"trained_at={_ms_to_hk(cache.trained_at_timestamp)} "
        f"threshold={cache.threshold:.2f} "
        f"val_wr={cache.threshold_val_win_rate:.4f} "
        f"val_signals={cache.threshold_val_signals}",
        flush=True,
    )
    return cache


def _ensure_csv(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=LIVE_COLUMNS).writeheader()


def _read_rows(path: Path = LIVE_CSV) -> list[dict]:
    _ensure_csv(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(rows: list[dict], path: Path = LIVE_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LIVE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in LIVE_COLUMNS})
    tmp.replace(path)


def _post_wechat(lines: list[str]) -> bool:
    webhook = os.getenv("WECHAT_WEBHOOK_URL", WECHAT_WEBHOOK_URL)
    if not webhook:
        print("[crypto_gat_live] WECHAT_WEBHOOK_URL is empty; skip notification", flush=True)
        return False
    payload = {"msgtype": "markdown", "markdown": {"content": "\n".join(lines)}}
    try:
        response = requests.post(webhook, json=payload, timeout=WECHAT_REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"[crypto_gat_live] wechat status={response.status_code} body={response.text}", flush=True)
            return False
        print("[crypto_gat_live] wechat sent", flush=True)
        return True
    except Exception as exc:
        print(f"[crypto_gat_live] wechat failed: {type(exc).__name__}: {exc}", flush=True)
        return False


def _validated_stats(rows: list[dict]) -> tuple[float | None, int, int]:
    total = 0
    wins = 0
    for row in rows:
        if row.get("strategy") != STRATEGY_NAME or row.get("validation_status") != "validated":
            continue
        if row.get("is_correct") not in {"True", "False", "true", "false", True, False}:
            continue
        total += 1
        wins += str(row.get("is_correct")).lower() == "true"
    return (wins / total if total else None), wins, total


def _direction_text(direction: str) -> str:
    return "看涨 UP" if direction == "up" else "看跌 DOWN" if direction == "down" else "不预测"


def _row_horizon_minutes(row: dict) -> int:
    signal_timestamp = int(float(row["signal_timestamp"]))
    validation_timestamp = int(float(row["validation_timestamp"]))
    return int(round((validation_timestamp - signal_timestamp) / 60_000))


def _send_prediction(row: dict, rows: list[dict]) -> None:
    accuracy, wins, total = _validated_stats(rows)
    rate_text = "暂无" if accuracy is None else f"{accuracy:.2%}"
    horizon_minutes = _row_horizon_minutes(row)
    _post_wechat(
        [
            f"【BTC/USDT CryptoGAT {horizon_minutes}m 实盘预测】",
            f"策略：{STRATEGY_NAME}",
            f"预测时长：{horizon_minutes} 分钟",
            f"方向：{_direction_text(row['direction'])}",
            f"置信度：{float(row['confidence']):.4f}",
            f"当前实盘累计准确率：{rate_text} ({wins}/{total})",
            f"信号价格：{float(row['signal_price']):.2f}",
            f"信号时间：{row['signal_time']}",
            f"验证时间：{row['validation_time']}",
            f"p_up：{float(row['up_signal_probability']):.4f}",
            f"p_down：{float(row['down_signal_probability']):.4f}",
            f"阈值：{float(row['threshold']):.2f}",
            f"配置：{row.get('reason', '')}",
        ]
    )


def _send_validation(row: dict, rows: list[dict]) -> None:
    accuracy, wins, total = _validated_stats(rows)
    rate_text = "暂无" if accuracy is None else f"{accuracy:.2%}"
    result = "正确" if str(row.get("is_correct")).lower() == "true" else "错误"
    horizon_minutes = _row_horizon_minutes(row)
    _post_wechat(
        [
            f"【BTC/USDT CryptoGAT {horizon_minutes}m 预测验证】",
            f"策略：{STRATEGY_NAME}",
            f"预测时长：{horizon_minutes} 分钟",
            f"预测ID：{row['prediction_id']}",
            f"预测方向：{_direction_text(row['direction'])}",
            f"实际方向：{_direction_text(row['actual_direction'])}",
            f"结果：{result}",
            f"信号价格：{float(row['signal_price']):.2f}",
            f"验证价格：{float(row['validation_price']):.2f}",
            f"信号时间：{row['signal_time']}",
            f"验证时间：{row['validation_time']}",
            f"当前实盘累计准确率：{rate_text} ({wins}/{total})",
        ]
    )


def _update_symbols(symbols: list[str], download_days: int, base_url: str) -> None:
    for symbol in symbols:
        _download_symbol(symbol, download_days, base_url)


def _refresh_recent_symbols(symbols: list[str], base_url: str) -> None:
    session = requests.Session()
    now_ms = int(time.time() * 1000)
    for symbol in symbols:
        path = _symbol_path(symbol)
        if not path.exists() or path.stat().st_size == 0:
            _download_symbol(symbol, 12, base_url)
            continue
        try:
            timestamps = pd.read_csv(path, usecols=["timestamp"])
        except Exception:
            _download_symbol(symbol, 12, base_url)
            continue
        timestamps["timestamp"] = pd.to_numeric(timestamps["timestamp"], errors="coerce")
        last_timestamp = int(timestamps["timestamp"].dropna().max())
        start_ms = last_timestamp + 60_000
        if start_ms > now_ms:
            print(f"[refresh] {symbol} no new rows", flush=True)
            continue
        params = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": start_ms,
            "endTime": now_ms,
            "limit": 10,
        }
        try:
            response = session.get(f"{base_url}/api/v3/klines", params=params, timeout=(5, 10))
            response.raise_for_status()
            batch = response.json()
        except Exception as exc:
            print(f"[refresh] {symbol} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not batch:
            print(f"[refresh] {symbol} no new rows", flush=True)
            continue
        fresh = pd.DataFrame(batch, columns=KLINE_COLUMNS).drop(columns=["ignore"])
        fresh["timestamp"] = pd.to_numeric(fresh["timestamp"], errors="coerce")
        fresh = fresh[fresh["timestamp"] > last_timestamp].dropna(subset=["timestamp"]).drop_duplicates("timestamp")
        if fresh.empty:
            print(f"[refresh] {symbol} no new rows", flush=True)
            continue
        fresh.to_csv(path, mode="a", header=False, index=False)
        print(
            f"[refresh] {symbol} appended={len(fresh)} "
            f"end={pd.to_datetime(fresh['timestamp'].max(), unit='ms', utc=True)}",
            flush=True,
        )


def _prepare_live_frames(
    *,
    symbols: list[str],
    train_days: int,
    inner_validation_days: int,
    bar_minutes: int,
    horizon_minutes: int,
    top_k: int,
    attention_temperature: float,
    feature_set: str,
    lookback_bars: int,
    xsec_features: bool,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict, int, int, int]:
    data, returns = _prepare_dataset(
        symbols,
        bar_minutes,
        horizon_minutes,
        as_of_ms=None,
        feature_set=feature_set,
        lookback_bars=lookback_bars,
        xsec_features=xsec_features,
    )
    btc_bars = _resample_bars(_load_symbol("BTCUSDT"), bar_minutes)[["timestamp", "close"]].drop_duplicates("timestamp")
    data = pd.merge(data, btc_bars, on="timestamp", how="left")
    latest = int(data["timestamp"].max())
    train_start = latest - (train_days + inner_validation_days) * 86_400_000
    inner_val_start = latest - inner_validation_days * 86_400_000
    graph_features, graph_meta = _build_graph_features(
        data,
        returns,
        symbols=symbols,
        train_start=train_start,
        graph_end=inner_val_start,
        top_k=top_k,
        attention_temperature=attention_temperature,
    )
    if model_name == "torch_cross_gat":
        suffixes = _common_node_suffixes(data, symbols)
        feature_cols = [f"{symbol}_{suffix}" for symbol in symbols for suffix in suffixes]
        label_cols = [f"{symbol}_future_return" for symbol in symbols if f"{symbol}_future_return" in data.columns]
        merged = data[["timestamp", "close", "future_return"] + feature_cols + label_cols].copy()
    else:
        merged = pd.merge(data[["timestamp", "close", "future_return"]], graph_features, on="timestamp", how="inner")
        feature_cols = [column for column in merged.columns if column not in {"timestamp", "close", "future_return"}]
    for column in feature_cols:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    train = merged[(merged["timestamp"] >= train_start) & (merged["timestamp"] < inner_val_start)].copy()
    val = merged[(merged["timestamp"] >= inner_val_start) & (merged["timestamp"] < latest)].copy()
    current = merged[merged["timestamp"] == latest].copy()
    for frame in (train, val, current):
        frame.dropna(subset=feature_cols, inplace=True)
    train = train.dropna(subset=["future_return"])
    val = val.dropna(subset=["future_return"])
    return data, returns, train, val, current, feature_cols, graph_meta, train_start, inner_val_start, latest


def _prepare_fast_current_frame(
    *,
    symbols: list[str],
    bar_minutes: int,
    horizon_minutes: int,
    feature_set: str,
    lookback_bars: int,
    xsec_features: bool,
    min_history_bars: int = 240,
) -> pd.DataFrame:
    if xsec_features:
        data, _, _, _, current, _, _, _, _, _ = _prepare_live_frames(
            symbols=symbols,
            train_days=1,
            inner_validation_days=1,
            bar_minutes=bar_minutes,
            horizon_minutes=horizon_minutes,
            top_k=1,
            attention_temperature=0.2,
            feature_set=feature_set,
            lookback_bars=lookback_bars,
            xsec_features=xsec_features,
            model_name="torch_cross_gat",
        )
        return current[["timestamp", "close"] + [c for c in current.columns if c not in {"timestamp", "close", "future_return"}]].copy()
    lags = [1, 2, 4, 8, 16]
    tail_bars = max(min_history_bars, int(lookback_bars) + max(lags) + int(np.ceil(horizon_minutes / max(bar_minutes, 1))) + 20)
    bars_by_symbol = {}
    latest_candidates = []
    for symbol in symbols:
        bars = _resample_bars(_load_symbol(symbol), bar_minutes).copy()
        if bars.empty:
            return pd.DataFrame()
        bars_by_symbol[symbol] = bars
        latest_candidates.append(int(bars["timestamp"].max()))
    latest_common_limit = min(latest_candidates)
    feature_parts = []
    btc_close = None
    for symbol in symbols:
        bars = bars_by_symbol[symbol]
        bars = bars[bars["timestamp"] <= latest_common_limit].tail(tail_bars).copy()
        if bars.empty:
            return pd.DataFrame()
        if symbol == "BTCUSDT":
            btc_close = bars[["timestamp", "close"]].copy()
        features = _node_features(bars, symbol, lags, feature_set, lookback_bars)
        feature_parts.append(features)
    merged = feature_parts[0]
    for part in feature_parts[1:]:
        merged = pd.merge(merged, part, on="timestamp", how="inner")
    merged = merged[merged["timestamp"] <= latest_common_limit].copy()
    if btc_close is None:
        return pd.DataFrame()
    merged = pd.merge(merged, btc_close, on="timestamp", how="left")
    merged = merged.replace([np.inf, -np.inf], np.nan).sort_values("timestamp")
    return merged.tail(max(tail_bars, 240)).copy()


def _train_torch_cross_gat_cache(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    returns: pd.DataFrame,
    symbols: list[str],
    train_start: int,
    inner_val_start: int,
    top_k: int,
    min_threshold_win_rate: float,
    min_threshold_signals_per_day: float,
    direction_filter: str,
    graph_meta: dict,
    train_days: int,
    inner_validation_days: int,
    bar_minutes: int,
    horizon_minutes: int,
    model_name: str,
) -> LiveModelCache | None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("torch is required for --model torch_cross_gat")
    suffixes = _common_node_suffixes(train, symbols)
    node_columns = [f"{symbol}_{suffix}" for symbol in symbols for suffix in suffixes]
    label_columns = [f"{symbol}_future_return" for symbol in symbols]
    train = train.dropna(subset=node_columns + label_columns).copy()
    val = val.dropna(subset=node_columns).copy()
    if len(train) < 300 or len(val) < 30 or not suffixes:
        return None
    x_train = np.stack([train[[f"{symbol}_{suffix}" for suffix in suffixes]].to_numpy(dtype=np.float32) for symbol in symbols], axis=1)
    x_val = np.stack([val[[f"{symbol}_{suffix}" for suffix in suffixes]].to_numpy(dtype=np.float32) for symbol in symbols], axis=1)
    y_train = (train[label_columns].to_numpy(dtype=float) > 0).astype(np.float32)
    if np.unique(y_train).size < 2:
        return None
    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    x_train = np.nan_to_num((x_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x_val = np.nan_to_num((x_val - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    adjacency = _adjacency_mask(returns, symbols, train_start, inner_val_start, top_k)
    device_name = "cpu"
    device = torch.device(device_name)
    torch.manual_seed(RANDOM_STATE)
    model = CrossAssetGraphAttentionClassifier(len(suffixes), 64, adjacency, 0.0).to(device)
    positives = float(y_train.sum())
    negatives = float(y_train.size - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.0001)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)
    model.train()
    for _ in range(30):
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
        for start in range(0, len(x_val), 256):
            batch = torch.from_numpy(x_val[start : start + 256]).to(device)
            outputs.append(torch.sigmoid(model(batch))[:, 0].detach().cpu().numpy())
    val_prob = np.concatenate(outputs) if outputs else np.array([], dtype=float)
    selected = _select_threshold(
        val_prob,
        val["future_return"].to_numpy(dtype=float),
        validation_days=inner_validation_days,
        min_win_rate=min_threshold_win_rate,
        min_signals_per_day=min_threshold_signals_per_day,
        direction_filter=direction_filter,
    )
    signals, _, val_wr_at_threshold = _eval_threshold(
        val_prob,
        val["future_return"].to_numpy(dtype=float),
        selected.threshold,
        direction_filter,
    )
    if selected.signals <= 0 or not np.isfinite(selected.win_rate):
        return None
    return LiveModelCache(
        model_name=model_name,
        trained_at_wall=time.time(),
        trained_at_timestamp=int(train["timestamp"].max()),
        feature_cols=node_columns,
        symbols=symbols,
        threshold=selected.threshold,
        threshold_val_win_rate=selected.win_rate if np.isfinite(selected.win_rate) else val_wr_at_threshold,
        threshold_val_signals=selected.signals or signals,
        threshold_val_wilson_lower=selected.wilson_lower,
        neighbors=graph_meta["neighbors"],
        attention_weights=graph_meta["attention_weights"],
        reason=f"crypto_gat_{model_name}_{bar_minutes}m_to_{horizon_minutes}m_top{top_k}_train{train_days}_val{inner_validation_days}_wr{int(min_threshold_win_rate * 100)}",
        model=model,
        suffixes=suffixes,
        mean=mean,
        std=std,
        batch_size=256,
        device_name=device_name,
    )


def _train_tree_cache(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    graph_meta: dict,
    train_days: int,
    inner_validation_days: int,
    bar_minutes: int,
    horizon_minutes: int,
    model_name: str,
    min_threshold_win_rate: float,
    min_threshold_signals_per_day: float,
    direction_filter: str,
    symbols: list[str],
) -> LiveModelCache | None:
    y_train = (train["future_return"] > 0).astype(int)
    if len(train) < 300 or len(val) < 30 or y_train.nunique() < 2:
        return None
    if model_name == "extra_xgb_avg":
        extra = _model("extra_trees")
        xgb = _model("xgboost")
        extra.fit(train[feature_cols], y_train)
        xgb.fit(train[feature_cols], y_train)
        val_prob = 0.5 * extra.predict_proba(val[feature_cols])[:, 1] + 0.5 * xgb.predict_proba(val[feature_cols])[:, 1]
        model = None
    else:
        model = _model(model_name)
        model.fit(train[feature_cols], y_train)
        extra = None
        xgb = None
        val_prob = model.predict_proba(val[feature_cols])[:, 1]
    selected = _select_threshold(
        val_prob,
        val["future_return"].to_numpy(dtype=float),
        validation_days=inner_validation_days,
        min_win_rate=min_threshold_win_rate,
        min_signals_per_day=min_threshold_signals_per_day,
        direction_filter=direction_filter,
    )
    signals, _, val_wr_at_threshold = _eval_threshold(
        val_prob,
        val["future_return"].to_numpy(dtype=float),
        selected.threshold,
        direction_filter,
    )
    if selected.signals <= 0 or not np.isfinite(selected.win_rate):
        return None
    return LiveModelCache(
        model_name=model_name,
        trained_at_wall=time.time(),
        trained_at_timestamp=int(train["timestamp"].max()),
        feature_cols=feature_cols,
        symbols=symbols,
        threshold=selected.threshold,
        threshold_val_win_rate=selected.win_rate if np.isfinite(selected.win_rate) else val_wr_at_threshold,
        threshold_val_signals=selected.signals or signals,
        threshold_val_wilson_lower=selected.wilson_lower,
        neighbors=graph_meta["neighbors"],
        attention_weights=graph_meta["attention_weights"],
        reason=f"crypto_gat_{model_name}_{bar_minutes}m_to_{horizon_minutes}m_top{top_k}_train{train_days}_val{inner_validation_days}_wr{int(min_threshold_win_rate * 100)}",
        model=model,
        extra_model=extra,
        xgb_model=xgb,
    )


def _train_live_cache(
    *,
    symbols: list[str],
    train_days: int,
    inner_validation_days: int,
    bar_minutes: int,
    horizon_minutes: int,
    top_k: int,
    attention_temperature: float,
    feature_set: str,
    lookback_bars: int,
    xsec_features: bool,
    model_name: str,
    min_threshold_win_rate: float,
    min_threshold_signals_per_day: float,
    direction_filter: str,
) -> tuple[LiveModelCache | None, pd.DataFrame]:
    data, returns, train, val, current, feature_cols, graph_meta, train_start, inner_val_start, _ = _prepare_live_frames(
        symbols=symbols,
        train_days=train_days,
        inner_validation_days=inner_validation_days,
        bar_minutes=bar_minutes,
        horizon_minutes=horizon_minutes,
        top_k=top_k,
        attention_temperature=attention_temperature,
        feature_set=feature_set,
        lookback_bars=lookback_bars,
        xsec_features=xsec_features,
        model_name=model_name,
    )
    if current.empty:
        return None, data
    if model_name == "torch_cross_gat":
        cache = _train_torch_cross_gat_cache(
            train=train,
            val=val,
            returns=returns,
            symbols=symbols,
            train_start=train_start,
            inner_val_start=inner_val_start,
            top_k=top_k,
            min_threshold_win_rate=min_threshold_win_rate,
            min_threshold_signals_per_day=min_threshold_signals_per_day,
            direction_filter=direction_filter,
            graph_meta=graph_meta,
            train_days=train_days,
            inner_validation_days=inner_validation_days,
            bar_minutes=bar_minutes,
            horizon_minutes=horizon_minutes,
            model_name=model_name,
        )
    else:
        cache = _train_tree_cache(
            train=train,
            val=val,
            feature_cols=feature_cols,
            graph_meta=graph_meta,
            train_days=train_days,
            inner_validation_days=inner_validation_days,
            bar_minutes=bar_minutes,
            horizon_minutes=horizon_minutes,
            model_name=model_name,
            min_threshold_win_rate=min_threshold_win_rate,
            min_threshold_signals_per_day=min_threshold_signals_per_day,
            direction_filter=direction_filter,
            symbols=symbols,
        )
    return cache, data


def _predict_with_cache(
    cache: LiveModelCache,
    current: pd.DataFrame,
    *,
    direction_filter: str,
    horizon_minutes: int,
) -> dict | None:
    if current.empty:
        return None
    if cache.model_name == "torch_cross_gat":
        if cache.model is None or cache.suffixes is None or cache.mean is None or cache.std is None:
            return None
        current = current.dropna(subset=cache.feature_cols).copy()
        if current.empty:
            return None
        x_current = np.stack(
            [current[[f"{symbol}_{suffix}" for suffix in cache.suffixes]].to_numpy(dtype=np.float32) for symbol in cache.symbols],
            axis=1,
        )
        x_current = np.nan_to_num((x_current - cache.mean) / cache.std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        device = torch.device(cache.device_name)
        cache.model.eval()
        with torch.no_grad():
            current_prob = float(torch.sigmoid(cache.model(torch.from_numpy(x_current).to(device)))[:, 0].detach().cpu().numpy()[0])
    elif cache.model_name == "extra_xgb_avg":
        current = current.dropna(subset=cache.feature_cols).copy()
        if current.empty or cache.extra_model is None or cache.xgb_model is None:
            return None
        current_prob = float(
            0.5 * cache.extra_model.predict_proba(current[cache.feature_cols])[:, 1][0]
            + 0.5 * cache.xgb_model.predict_proba(current[cache.feature_cols])[:, 1][0]
        )
    else:
        current = current.dropna(subset=cache.feature_cols).copy()
        if current.empty or cache.model is None:
            return None
        current_prob = float(cache.model.predict_proba(current[cache.feature_cols])[:, 1][0])
    direction = "no_trade"
    if direction_filter in {"both", "up"} and current_prob >= cache.threshold:
        direction = "up"
    elif direction_filter in {"both", "down"} and current_prob <= 1.0 - cache.threshold:
        direction = "down"
    latest = int(current.iloc[0]["timestamp"])
    return {
        "timestamp": latest,
        "price": float(current.iloc[0]["close"]),
        "direction": direction,
        "confidence": current_prob if direction == "up" else 1.0 - current_prob if direction == "down" else max(current_prob, 1.0 - current_prob),
        "up_signal_probability": current_prob,
        "down_signal_probability": 1.0 - current_prob,
        "direction_edge": current_prob - (1.0 - current_prob),
        "threshold": cache.threshold,
        "threshold_val_win_rate": cache.threshold_val_win_rate,
        "threshold_val_signals": cache.threshold_val_signals,
        "threshold_val_wilson_lower": cache.threshold_val_wilson_lower,
        "neighbors": cache.neighbors,
        "attention_weights": cache.attention_weights,
        "horizon_minutes": horizon_minutes,
        "reason": cache.reason,
    }


def _latest_completed_prediction(
    *,
    symbols: list[str],
    train_days: int,
    inner_validation_days: int,
    bar_minutes: int,
    horizon_minutes: int,
    top_k: int,
    attention_temperature: float,
    feature_set: str,
    lookback_bars: int,
    xsec_features: bool,
    model_name: str,
    min_threshold_win_rate: float,
    min_threshold_signals_per_day: float,
    direction_filter: str,
) -> tuple[dict | None, pd.DataFrame]:
    data, returns = _prepare_dataset(
        symbols,
        bar_minutes,
        horizon_minutes,
        as_of_ms=None,
        feature_set=feature_set,
        lookback_bars=lookback_bars,
        xsec_features=xsec_features,
    )
    btc_bars = _resample_bars(_load_symbol("BTCUSDT"), bar_minutes)[["timestamp", "close"]].drop_duplicates("timestamp")
    data = pd.merge(data, btc_bars, on="timestamp", how="left")
    latest = int(data["timestamp"].max())
    train_start = latest - (train_days + inner_validation_days) * 86_400_000
    inner_val_start = latest - inner_validation_days * 86_400_000
    graph_features, graph_meta = _build_graph_features(
        data,
        returns,
        symbols=symbols,
        train_start=train_start,
        graph_end=inner_val_start,
        top_k=top_k,
        attention_temperature=attention_temperature,
    )
    if model_name == "torch_cross_gat":
        suffixes = _common_node_suffixes(data, symbols)
        feature_cols = [f"{symbol}_{suffix}" for symbol in symbols for suffix in suffixes]
        label_cols = [f"{symbol}_future_return" for symbol in symbols if f"{symbol}_future_return" in data.columns]
        merged = data[["timestamp", "close", "future_return"] + feature_cols + label_cols].copy()
    else:
        merged = pd.merge(data[["timestamp", "close", "future_return"]], graph_features, on="timestamp", how="inner")
        feature_cols = [column for column in merged.columns if column not in {"timestamp", "close", "future_return"}]
    train = merged[(merged["timestamp"] >= train_start) & (merged["timestamp"] < inner_val_start)].copy()
    val = merged[(merged["timestamp"] >= inner_val_start) & (merged["timestamp"] < latest)].copy()
    current = merged[merged["timestamp"] == latest].copy()
    for frame in (train, val, current):
        frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
        frame.dropna(subset=feature_cols, inplace=True)
    train = train.dropna(subset=["future_return"])
    val = val.dropna(subset=["future_return"])
    if len(train) < 300 or len(val) < 30 or current.empty:
        return None, data
    y_train = (train["future_return"] > 0).astype(int)
    if y_train.nunique() < 2:
        return None, data
    if model_name == "torch_cross_gat":
        combined_predict = pd.concat([val, current], ignore_index=True)
        combined_prob = _torch_cross_asset_gat_probabilities(
            train,
            combined_predict,
            returns_frame=returns,
            symbols=symbols,
            train_start=train_start,
            graph_end=inner_val_start,
            top_k=top_k,
            epochs=30,
            hidden_size=64,
            learning_rate=0.00025,
            weight_decay=0.0001,
            batch_size=256,
            dropout=0.0,
            device_name="cpu",
        )
        if len(combined_prob) != len(combined_predict):
            return None, data
        val_prob = combined_prob[: len(val)]
        current_prob_array = combined_prob[len(val) :]
    else:
        try:
            val_prob, current_prob_array = _fit_predict_tree_probability(model_name, train, val, current, feature_cols)
        except ValueError:
            return None, data
    selected = _select_threshold(
        val_prob,
        val["future_return"].to_numpy(dtype=float),
        validation_days=inner_validation_days,
        min_win_rate=min_threshold_win_rate,
        min_signals_per_day=min_threshold_signals_per_day,
        direction_filter=direction_filter,
    )
    current_prob = float(current_prob_array[0])
    signals, wins, val_wr_at_threshold = _eval_threshold(
        val_prob,
        val["future_return"].to_numpy(dtype=float),
        selected.threshold,
        direction_filter,
    )
    direction = "no_trade"
    if direction_filter in {"both", "up"} and current_prob >= selected.threshold:
        direction = "up"
    elif direction_filter in {"both", "down"} and current_prob <= 1.0 - selected.threshold:
        direction = "down"
    decision = {
        "timestamp": latest,
        "price": float(current.iloc[0]["close"]),
        "direction": direction,
        "confidence": current_prob if direction == "up" else 1.0 - current_prob if direction == "down" else max(current_prob, 1.0 - current_prob),
        "up_signal_probability": current_prob,
        "down_signal_probability": 1.0 - current_prob,
        "direction_edge": current_prob - (1.0 - current_prob),
        "threshold": selected.threshold,
        "threshold_val_win_rate": selected.win_rate if np.isfinite(selected.win_rate) else val_wr_at_threshold,
        "threshold_val_signals": selected.signals or signals,
        "threshold_val_wilson_lower": selected.wilson_lower,
        "neighbors": graph_meta["neighbors"],
        "attention_weights": graph_meta["attention_weights"],
        "horizon_minutes": horizon_minutes,
        "reason": f"crypto_gat_{model_name}_{bar_minutes}m_to_{horizon_minutes}m_top{top_k}_train{train_days}_val{inner_validation_days}_wr{int(min_threshold_win_rate * 100)}",
    }
    return decision, data


def _validate_due(data: pd.DataFrame) -> None:
    rows = _read_rows()
    if not rows:
        return
    close_by_timestamp = data.dropna(subset=["close"]).set_index("timestamp")["close"]
    changed = False
    for row in rows:
        if row.get("validation_status") != "pending":
            continue
        validation_timestamp = int(float(row["validation_timestamp"]))
        if validation_timestamp not in close_by_timestamp.index:
            continue
        signal_price = float(row["signal_price"])
        validation_price = float(close_by_timestamp.loc[validation_timestamp])
        if validation_price > signal_price:
            actual_direction = "up"
        elif validation_price < signal_price:
            actual_direction = "down"
        else:
            actual_direction = "flat"
        row["actual_direction"] = actual_direction
        row["validation_price"] = f"{validation_price:.8f}"
        row["is_correct"] = str(actual_direction == row["direction"])
        row["validation_status"] = "validated"
        changed = True
        _write_rows(rows)
        _send_validation(row, rows)
    if changed:
        _write_rows(rows)


def _record_prediction(decision: dict) -> None:
    rows = _read_rows()
    prediction_id = f"{STRATEGY_NAME}-{int(decision['timestamp'])}"
    if any(row.get("prediction_id") == prediction_id for row in rows):
        print(f"[crypto_gat_live] already recorded {prediction_id}", flush=True)
        return
    signal_timestamp = int(decision["timestamp"])
    horizon_minutes = int(decision.get("horizon_minutes", PREDICT_HORIZON_MINUTES))
    validation_timestamp = signal_timestamp + horizon_minutes * 60_000
    row = {
        "prediction_id": prediction_id,
        "strategy": STRATEGY_NAME,
        "signal_timestamp": signal_timestamp,
        "signal_time": _ms_to_hk(signal_timestamp),
        "validation_timestamp": validation_timestamp,
        "validation_time": _ms_to_hk(validation_timestamp),
        "signal_price": f"{float(decision['price']):.8f}",
        "direction": decision["direction"],
        "confidence": f"{float(decision['confidence']):.8f}",
        "up_signal_probability": f"{float(decision['up_signal_probability']):.8f}",
        "down_signal_probability": f"{float(decision['down_signal_probability']):.8f}",
        "direction_edge": f"{float(decision['direction_edge']):.8f}",
        "threshold": f"{float(decision['threshold']):.8f}",
        "threshold_val_win_rate": f"{float(decision['threshold_val_win_rate']):.8f}" if np.isfinite(float(decision["threshold_val_win_rate"])) else "",
        "threshold_val_signals": int(decision["threshold_val_signals"]),
        "threshold_val_wilson_lower": f"{float(decision['threshold_val_wilson_lower']):.8f}",
        "neighbors": ",".join(decision["neighbors"]),
        "attention_weights": json.dumps(decision["attention_weights"], ensure_ascii=False, sort_keys=True),
        "reason": decision.get("reason", "crypto_gat"),
        "validation_status": "pending",
        "actual_direction": "",
        "validation_price": "",
        "is_correct": "",
        "notified": "true",
    }
    rows.append(row)
    _write_rows(rows)
    _send_prediction(row, rows)


def _save_state(payload: dict) -> None:
    LIVE_STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_loop(args: argparse.Namespace) -> None:
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    if "BTCUSDT" not in symbols:
        symbols.insert(0, "BTCUSDT")
    cache: LiveModelCache | None = _load_model_cache(args)
    last_data_refresh_wall = 0.0
    last_retrain_wall = 0.0
    last_processed_timestamp: int | None = None
    full_refresh_done = False
    retrain_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending_retrain: concurrent.futures.Future | None = None

    def submit_retrain() -> concurrent.futures.Future:
        print("[crypto_gat_live] async retraining cached model", flush=True)
        return retrain_executor.submit(
            _train_live_cache,
            symbols=symbols,
            train_days=args.train_days,
            inner_validation_days=args.inner_validation_days,
            bar_minutes=args.bar_minutes,
            horizon_minutes=args.horizon_minutes,
            top_k=args.top_k,
            attention_temperature=args.attention_temperature,
            feature_set=args.feature_set,
            lookback_bars=args.lookback_bars,
            xsec_features=args.xsec_features,
            model_name=args.model,
            min_threshold_win_rate=args.min_threshold_win_rate,
            min_threshold_signals_per_day=args.min_threshold_signals_per_day,
            direction_filter=args.direction_filter,
        )

    try:
        while True:
            loop_started = time.time()
            try:
                if pending_retrain is not None and pending_retrain.done():
                    next_cache, _ = pending_retrain.result()
                    pending_retrain = None
                    if next_cache is not None:
                        cache = next_cache
                        last_retrain_wall = time.time()
                        _save_model_cache(cache, args)
                        print(
                            "[crypto_gat_live] cached model ready "
                            f"trained_at={_ms_to_hk(cache.trained_at_timestamp)} "
                            f"threshold={cache.threshold:.2f} "
                            f"val_wr={cache.threshold_val_win_rate if np.isfinite(cache.threshold_val_win_rate) else 'nan'} "
                            f"val_signals={cache.threshold_val_signals}",
                            flush=True,
                        )
                    else:
                        print("[crypto_gat_live] cached model retrain skipped; insufficient data", flush=True)
                        last_retrain_wall = time.time()

                if loop_started - last_data_refresh_wall >= args.data_refresh_seconds:
                    if full_refresh_done and args.fast_incremental_refresh:
                        _refresh_recent_symbols(symbols, args.binance_base_url)
                    else:
                        _update_symbols(symbols, args.download_days, args.binance_base_url)
                        full_refresh_done = True
                    last_data_refresh_wall = time.time()

                if cache is None:
                    print("[crypto_gat_live] initial cached model training", flush=True)
                    next_cache, data = _train_live_cache(
                        symbols=symbols,
                        train_days=args.train_days,
                        inner_validation_days=args.inner_validation_days,
                        bar_minutes=args.bar_minutes,
                        horizon_minutes=args.horizon_minutes,
                        top_k=args.top_k,
                        attention_temperature=args.attention_temperature,
                        feature_set=args.feature_set,
                        lookback_bars=args.lookback_bars,
                        xsec_features=args.xsec_features,
                        model_name=args.model,
                        min_threshold_win_rate=args.min_threshold_win_rate,
                        min_threshold_signals_per_day=args.min_threshold_signals_per_day,
                        direction_filter=args.direction_filter,
                    )
                    if next_cache is not None:
                        cache = next_cache
                        last_retrain_wall = time.time()
                        _save_model_cache(cache, args)
                        print(
                            "[crypto_gat_live] cached model ready "
                            f"trained_at={_ms_to_hk(cache.trained_at_timestamp)} "
                            f"threshold={cache.threshold:.2f} "
                            f"val_wr={cache.threshold_val_win_rate if np.isfinite(cache.threshold_val_win_rate) else 'nan'} "
                            f"val_signals={cache.threshold_val_signals}",
                            flush=True,
                        )
                    else:
                        print("[crypto_gat_live] initial cached model unavailable", flush=True)

                if cache is not None and args.fast_inference:
                    data = _prepare_fast_current_frame(
                        symbols=symbols,
                        bar_minutes=args.bar_minutes,
                        horizon_minutes=args.horizon_minutes,
                        feature_set=args.feature_set,
                        lookback_bars=args.lookback_bars,
                        xsec_features=args.xsec_features,
                    )
                    current = data.tail(1).copy()
                else:
                    data, _, _, _, current, _, _, _, _, _ = _prepare_live_frames(
                        symbols=symbols,
                        train_days=args.train_days,
                        inner_validation_days=args.inner_validation_days,
                        bar_minutes=args.bar_minutes,
                        horizon_minutes=args.horizon_minutes,
                        top_k=args.top_k,
                        attention_temperature=args.attention_temperature,
                        feature_set=args.feature_set,
                        lookback_bars=args.lookback_bars,
                        xsec_features=args.xsec_features,
                        model_name=args.model,
                    )
                if data is not None:
                    _validate_due(data)

                decision = _predict_with_cache(
                    cache,
                    current,
                    direction_filter=args.direction_filter,
                    horizon_minutes=args.horizon_minutes,
                ) if cache is not None else None
                if decision is None:
                    print("[crypto_gat_live] no decision; insufficient data or model unavailable", flush=True)
                elif last_processed_timestamp == int(decision["timestamp"]):
                    print(
                        "[crypto_gat_live] latest bar already processed "
                        f"time={_ms_to_hk(int(decision['timestamp']))}",
                        flush=True,
                    )
                else:
                    last_processed_timestamp = int(decision["timestamp"])
                    print(
                        "[crypto_gat_live] decision "
                        f"time={_ms_to_hk(int(decision['timestamp']))} direction={decision['direction']} "
                        f"p_up={decision['up_signal_probability']:.4f} threshold={decision['threshold']:.2f} "
                        f"neighbors={decision['neighbors']}",
                        flush=True,
                    )
                    _save_state({"last_decision": decision, "updated_at": _ms_to_hk(int(time.time() * 1000))})
                    if decision["direction"] in {"up", "down"}:
                        _record_prediction(decision)

                retrain_due = cache is not None and loop_started - last_retrain_wall >= args.retrain_seconds
                if retrain_due and pending_retrain is None:
                    pending_retrain = submit_retrain()
                elif retrain_due and pending_retrain is not None:
                    print("[crypto_gat_live] retrain already running; keep using previous cached model", flush=True)

                if args.once:
                    return
            except Exception as exc:
                print(f"[crypto_gat_live] error: {type(exc).__name__}: {exc}", flush=True)
                if args.once:
                    raise
            elapsed = time.time() - loop_started
            time.sleep(max(1.0, args.interval_seconds - elapsed))
    finally:
        retrain_executor.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone CryptoGAT-style BTC realtime predictor.")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT")
    parser.add_argument("--download-days", type=int, default=180)
    parser.add_argument("--binance-base-url", default="https://data-api.binance.vision")
    parser.add_argument("--train-days", type=int, default=20)
    parser.add_argument("--inner-validation-days", type=int, default=5)
    parser.add_argument("--bar-minutes", type=int, default=1)
    parser.add_argument("--horizon-minutes", type=int, default=PREDICT_HORIZON_MINUTES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--attention-temperature", type=float, default=0.2)
    parser.add_argument("--feature-set", choices=["compact", "fgat"], default="compact")
    parser.add_argument("--lookback-bars", type=int, default=30)
    parser.add_argument("--xsec-features", action="store_true")
    parser.add_argument("--model", choices=["extra_trees", "lgb", "xgboost", "extra_xgb_avg", "torch_cross_gat"], default="extra_trees")
    parser.add_argument("--min-threshold-win-rate", type=float, default=0.65)
    parser.add_argument("--min-threshold-signals-per-day", type=float, default=0.05)
    parser.add_argument("--direction-filter", choices=["both", "up", "down"], default="both")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--data-refresh-seconds", type=int, default=60)
    parser.add_argument("--retrain-seconds", type=int, default=1800)
    parser.add_argument("--fast-incremental-refresh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()
