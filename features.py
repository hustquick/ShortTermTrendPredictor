# features.py

import numpy as np
import pandas as pd

from config import (
    LABEL_NEUTRAL_THRESHOLD,
    PREDICT_HORIZON_MS,
    USE_LABEL_NEUTRAL_ZONE,
)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series):
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)

    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal

    return macd, signal, hist


def _kdj(df: pd.DataFrame, window: int = 9):
    low_min = df["low"].rolling(window).min()
    high_max = df["high"].rolling(window).max()

    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-12) * 100

    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d

    return k, d, j


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建模型特征。

    只使用当前及过去数据，不使用未来数据。
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # =========================
    # 收益率特征
    # =========================

    for n in [1, 2, 3, 5, 10, 15, 20, 30]:
        df[f"ret_{n}"] = close.pct_change(n)

    # =========================
    # 均线与 EMA 特征
    # =========================

    for n in [5, 10, 20, 30, 60]:
        ma = close.rolling(n).mean()
        ema = _ema(close, n)

        df[f"ma_{n}_ratio"] = close / (ma + 1e-12) - 1
        df[f"ema_{n}_ratio"] = close / (ema + 1e-12) - 1
        df[f"ema_{n}_slope"] = ema.pct_change()

    df["ema_5_20_diff"] = _ema(close, 5) / (_ema(close, 20) + 1e-12) - 1
    df["ema_10_30_diff"] = _ema(close, 10) / (_ema(close, 30) + 1e-12) - 1
    df["ema_20_60_diff"] = _ema(close, 20) / (_ema(close, 60) + 1e-12) - 1

    # =========================
    # MACD
    # =========================

    macd, macd_signal, macd_hist = _macd(close)

    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["macd_hist_diff"] = macd_hist.diff()

    # =========================
    # RSI
    # =========================

    df["rsi_6"] = _rsi(close, 6)
    df["rsi_14"] = _rsi(close, 14)

    # =========================
    # KDJ
    # =========================

    k, d, j = _kdj(df)

    df["kdj_k"] = k
    df["kdj_d"] = d
    df["kdj_j"] = j

    # =========================
    # 布林带
    # =========================

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()

    upper = mid + 2 * std
    lower = mid - 2 * std

    df["boll_position"] = (close - lower) / (upper - lower + 1e-12)
    df["boll_width"] = (upper - lower) / (mid + 1e-12)

    # =========================
    # 波动率
    # =========================

    df["volatility_5"] = close.pct_change().rolling(5).std()
    df["volatility_10"] = close.pct_change().rolling(10).std()
    df["volatility_30"] = close.pct_change().rolling(30).std()

    df["atr_14"] = _atr(df, 14) / (close + 1e-12)

    # =========================
    # K 线结构
    # =========================

    candle_range = high - low
    body = (close - open_).abs()

    df["body_ratio"] = body / (candle_range + 1e-12)
    df["upper_shadow_ratio"] = (high - np.maximum(open_, close)) / (candle_range + 1e-12)
    df["lower_shadow_ratio"] = (np.minimum(open_, close) - low) / (candle_range + 1e-12)
    df["close_position"] = (close - low) / (candle_range + 1e-12)

    # =========================
    # 成交量特征
    # =========================

    for n in [5, 10, 20, 30]:
        vol_ma = volume.rolling(n).mean()
        df[f"volume_ratio_{n}"] = volume / (vol_ma + 1e-12)

    vol_mean = volume.rolling(30).mean()
    vol_std = volume.rolling(30).std()

    df["volume_zscore"] = (volume - vol_mean) / (vol_std + 1e-12)
    df["volume_change"] = volume.pct_change()

    # =========================
    # 量价相关
    # =========================

    df["price_volume_corr_10"] = close.pct_change().rolling(10).corr(volume.pct_change())

    # =========================
    # 主动买入成交结构
    # =========================

    df["taker_buy_ratio"] = df["taker_buy_base_volume"] / (df["volume"] + 1e-12)

    df["taker_buy_ratio_change"] = df["taker_buy_ratio"].diff()

    df["taker_buy_ratio_ma_5"] = df["taker_buy_ratio"].rolling(5).mean()
    df["taker_buy_ratio_ma_10"] = df["taker_buy_ratio"].rolling(10).mean()

    df["taker_buy_ratio_diff_5_10"] = (
        df["taker_buy_ratio_ma_5"] - df["taker_buy_ratio_ma_10"]
    )

    df["avg_trade_size"] = df["volume"] / (df["number_of_trades"] + 1e-12)

    df["trade_count_ratio_10"] = df["number_of_trades"] / (
        df["number_of_trades"].rolling(10).mean() + 1e-12
    )

    df["quote_volume_ratio_10"] = df["quote_asset_volume"] / (
        df["quote_asset_volume"].rolling(10).mean() + 1e-12
    )

    # =========================
    # 简单趋势一致性
    # =========================

    df["trend_5"] = np.sign(close - close.shift(5))
    df["trend_10"] = np.sign(close - close.shift(10))
    df["trend_15"] = np.sign(close - close.shift(15))

    df["trend_agreement"] = (
        df["trend_5"] + df["trend_10"] + df["trend_15"]
    ) / 3.0

    return df


def add_future_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建固定终点方向标签。

    核心目标：
    判断未来第 10 分钟 close 是否高于当前 close。

    默认：
    - future_close > current_close → label=1
    - future_close <= current_close → label=0

    如果 USE_LABEL_NEUTRAL_ZONE=True：
    - future_return > +LABEL_NEUTRAL_THRESHOLD → label=1
    - future_return < -LABEL_NEUTRAL_THRESHOLD → label=0
    - 中间样本 label=NaN，训练时删除
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    close_by_timestamp = df.set_index("timestamp")["close"]
    future_timestamp = df["timestamp"] + PREDICT_HORIZON_MS

    # 标签必须通过 13 位毫秒时间戳精确匹配未来 10 分钟 K 线，不能依赖行号偏移。
    df["future_price"] = future_timestamp.map(close_by_timestamp)
    df["future_return"] = df["future_price"] / df["close"] - 1

    if USE_LABEL_NEUTRAL_ZONE:
        df["label"] = np.nan
        df.loc[df["future_return"] > LABEL_NEUTRAL_THRESHOLD, "label"] = 1
        df.loc[df["future_return"] < -LABEL_NEUTRAL_THRESHOLD, "label"] = 0
    else:
        df["label"] = np.nan
        valid_future = df["future_price"].notna()
        df.loc[valid_future, "label"] = (
            df.loc[valid_future, "future_price"] > df.loc[valid_future, "close"]
        ).astype(int)

    return df


def add_dual_future_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建双方向子模型标签。

    up_label:
    - future_return > +LABEL_NEUTRAL_THRESHOLD → 1
    - 其他有未来价格的样本 → 0

    down_label:
    - future_return < -LABEL_NEUTRAL_THRESHOLD → 1
    - 其他有未来价格的样本 → 0

    与单模型不同，中性样本不会被删除，而是作为两个方向模型的负类。
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    close_by_timestamp = df.set_index("timestamp")["close"]
    future_timestamp = df["timestamp"] + PREDICT_HORIZON_MS

    df["future_price"] = future_timestamp.map(close_by_timestamp)
    df["future_return"] = df["future_price"] / df["close"] - 1

    valid_future = df["future_price"].notna()

    df["up_label"] = np.nan
    df["down_label"] = np.nan

    df.loc[valid_future, "up_label"] = (
        df.loc[valid_future, "future_return"] > LABEL_NEUTRAL_THRESHOLD
    ).astype(int)
    df.loc[valid_future, "down_label"] = (
        df.loc[valid_future, "future_return"] < -LABEL_NEUTRAL_THRESHOLD
    ).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame):
    """
    返回模型输入特征列。
    """
    exclude = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "close_time",
        "future_price",
        "future_return",
        "label",
        "up_label",
        "down_label",
    }

    return [col for col in df.columns if col not in exclude]


def make_train_dataset(df: pd.DataFrame):
    """
    构建单模型训练数据。

    返回：
    - X
    - y
    - data
    - feature_cols
    """
    df_feat = build_features(df)
    df_labeled = add_future_label(df_feat)

    feature_cols = get_feature_columns(df_labeled)

    data = df_labeled.dropna(
        subset=feature_cols + ["label"]
    ).copy()

    data = data.reset_index(drop=True)

    X = data[feature_cols]
    y = data["label"].astype(int)

    return X, y, data, feature_cols


def make_dual_train_dataset(df: pd.DataFrame):
    """
    构建双方向模型训练数据。

    返回：
    - X
    - y_up
    - y_down
    - data
    - feature_cols
    """
    df_feat = build_features(df)
    df_labeled = add_dual_future_labels(df_feat)

    feature_cols = get_feature_columns(df_labeled)

    data = df_labeled.dropna(
        subset=feature_cols + ["up_label", "down_label"]
    ).copy()

    data = data.reset_index(drop=True)

    X = data[feature_cols]
    y_up = data["up_label"].astype(int)
    y_down = data["down_label"].astype(int)

    return X, y_up, y_down, data, feature_cols


def make_realtime_features(df: pd.DataFrame, feature_cols):
    """
    实时预测特征构建。

    只使用当前及过去 K 线。
    """
    df_feat = build_features(df)
    df_feat = df_feat.dropna(subset=feature_cols).copy()

    if df_feat.empty:
        return None, None

    latest_row = df_feat.iloc[[-1]].copy()
    X_latest = latest_row[feature_cols]

    return X_latest, latest_row
