# data_download.py

import os
import time
import warnings
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import urllib3

from config import (
    BINANCE_BASE_URL,
    BINANCE_KLINES_ENDPOINT,
    BINANCE_LIMIT,
    BINANCE_VISION_BASE_URL,
    HISTORY_CSV,
    INTERVAL,
    INTERVAL_MS,
    OKX_BAR,
    OKX_BASE_URL,
    OKX_CANDLES_ENDPOINT,
    OKX_HISTORY_CANDLES_ENDPOINT,
    OKX_INST_ID,
    REQUEST_TIMEOUT,
    REQUEST_VERIFY_SSL,
    SYMBOL,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


KLINE_COLUMNS = [
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
]


def utc_ms_now() -> int:
    """返回当前 UTC 13 位毫秒时间戳。"""
    return int(time.time() * 1000)


def ms_to_beijing_time(ms: int) -> str:
    """13 位毫秒 UTC 时间戳转东八区字符串。"""
    if ms < 10_000_000_000:
        raise ValueError(f"时间戳异常，疑似 10 位秒级时间戳：{ms}")

    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    bj_dt = dt.astimezone(timezone(timedelta(hours=8)))
    return bj_dt.strftime("%Y-%m-%d %H:%M:%S")


def beijing_time_to_ms(time_str: str) -> int:
    """东八区时间字符串转 UTC 13 位毫秒时间戳。"""
    bj_tz = timezone(timedelta(hours=8))
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=bj_tz)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def validate_ms_timestamp(ms: int, name: str = "timestamp"):
    """检查是否为 13 位毫秒时间戳。"""
    if int(ms) < 10_000_000_000:
        raise ValueError(f"禁止使用 10 位秒级时间戳：{name}={ms}")


def _standard_headers() -> dict:
    api_key = os.getenv("BINANCE_API_KEY", "")
    _api_secret = os.getenv("BINANCE_API_SECRET", "")
    return {
        "X-MBX-APIKEY": api_key,
        "User-Agent": "ShortTermTrendPredictor/1.0",
    }


def _request_binance_klines(base_url: str, start_ms=None, end_ms=None, limit=BINANCE_LIMIT):
    """请求币安兼容 K 线数据。"""
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": limit,
    }

    if start_ms is not None:
        validate_ms_timestamp(start_ms, "start_ms")
        params["startTime"] = int(start_ms)

    if end_ms is not None:
        validate_ms_timestamp(end_ms, "end_ms")
        params["endTime"] = int(end_ms)

    url = base_url + BINANCE_KLINES_ENDPOINT

    response = requests.get(
        url,
        headers=_standard_headers(),
        params=params,
        timeout=REQUEST_TIMEOUT,
        verify=REQUEST_VERIFY_SSL,
    )

    response.raise_for_status()
    return response.json()


def _request_okx_candles(endpoint: str, cursor_ms: int | None, limit: int = 300):
    """请求 OKX BTC-USDT 1m K 线。"""
    params = {
        "instId": OKX_INST_ID,
        "bar": OKX_BAR,
        "limit": min(int(limit), 300),
    }

    if cursor_ms is not None:
        validate_ms_timestamp(cursor_ms, "okx_after")
        params["after"] = int(cursor_ms)

    response = requests.get(
        OKX_BASE_URL + endpoint,
        headers=_standard_headers(),
        params=params,
        timeout=REQUEST_TIMEOUT,
        verify=REQUEST_VERIFY_SSL,
    )
    response.raise_for_status()

    payload = response.json()
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX 返回错误: {payload}")

    return payload.get("data", [])


def klines_to_dataframe(raw_klines) -> pd.DataFrame:
    """将币安原始 K 线转换为标准 DataFrame。"""
    raw_columns = [
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

    df = pd.DataFrame(raw_klines, columns=raw_columns)

    if df.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    numeric_cols = [
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
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "close"]).copy()

    df["timestamp"] = df["timestamp"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")

    if (df["timestamp"] < 10_000_000_000).any():
        raise ValueError("检测到 10 位秒级时间戳，已终止。")

    df = df[KLINE_COLUMNS].copy()

    df = clean_kline_dataframe(df)

    return df


def okx_klines_to_dataframe(raw_klines) -> pd.DataFrame:
    """将 OKX 原始 K 线转换为标准 DataFrame。"""
    rows = []

    for row in raw_klines:
        if len(row) < 6:
            continue

        timestamp = int(row[0])
        validate_ms_timestamp(timestamp, "okx_timestamp")

        rows.append(
            {
                "timestamp": timestamp,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "quote_asset_volume": row[7] if len(row) > 7 else row[6],
                "number_of_trades": 0,
                "taker_buy_base_volume": 0,
                "taker_buy_quote_volume": 0,
                "close_time": timestamp + INTERVAL_MS - 1,
            }
        )

    return clean_kline_dataframe(pd.DataFrame(rows))


def clean_kline_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """清洗标准 K 线 DataFrame。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = df.copy()

    for col in KLINE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[KLINE_COLUMNS].copy()

    numeric_cols = KLINE_COLUMNS

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()

    df["timestamp"] = df["timestamp"].astype("int64")

    if "close_time" in df.columns:
        df["close_time"] = df["close_time"].fillna(df["timestamp"] + INTERVAL_MS - 1)
        df["close_time"] = df["close_time"].astype("int64")

    if (df["timestamp"] < 10_000_000_000).any():
        raise ValueError("历史数据中检测到 10 位秒级时间戳。")

    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def load_history_csv(path=HISTORY_CSV) -> pd.DataFrame:
    """读取本地历史 K 线缓存。"""
    if not path.exists():
        return pd.DataFrame(columns=KLINE_COLUMNS)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = clean_kline_dataframe(df)
    return df


def save_history_csv(df: pd.DataFrame, path=HISTORY_CSV):
    """保存本地历史 K 线缓存。"""
    df = clean_kline_dataframe(df)
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def merge_history(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """合并本地历史和新下载数据。"""
    old_df = clean_kline_dataframe(old_df)
    new_df = clean_kline_dataframe(new_df)

    if old_df.empty and new_df.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = clean_kline_dataframe(merged)

    return merged


def _fetch_binance_like_between(source_name: str, base_url: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """从币安兼容接口按毫秒区间拉取 K 线。"""
    all_rows = []
    current_start = int(start_ms)

    while current_start <= end_ms:
        raw = _request_binance_klines(
            base_url=base_url,
            start_ms=current_start,
            end_ms=end_ms,
            limit=BINANCE_LIMIT,
        )

        if not raw:
            break

        all_rows.extend(raw)

        last_open_time = int(raw[-1][0])
        next_start = last_open_time + INTERVAL_MS

        if next_start <= current_start:
            break

        current_start = next_start

        if len(raw) < BINANCE_LIMIT:
            break

    df = klines_to_dataframe(all_rows)
    print(f"[数据源] {source_name} 获取 K 线 {len(df)} 条。")
    return df


def _fetch_okx_between(start_ms: int, end_ms: int) -> pd.DataFrame:
    """从 OKX 按毫秒区间拉取 BTC-USDT 1m K 线。"""
    all_rows = []
    cursor_ms = int(end_ms) + INTERVAL_MS

    while True:
        raw = _request_okx_candles(
            endpoint=OKX_HISTORY_CANDLES_ENDPOINT,
            cursor_ms=cursor_ms,
            limit=300,
        )

        if not raw:
            raw = _request_okx_candles(
                endpoint=OKX_CANDLES_ENDPOINT,
                cursor_ms=cursor_ms,
                limit=300,
            )

        if not raw:
            break

        all_rows.extend(raw)
        timestamps = [int(row[0]) for row in raw]
        oldest = min(timestamps)

        if oldest <= start_ms:
            break

        next_cursor = oldest
        if next_cursor >= cursor_ms:
            break

        cursor_ms = next_cursor

    df = okx_klines_to_dataframe(all_rows)
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)].copy()
    df = clean_kline_dataframe(df)
    print(f"[数据源] OKX 获取 K 线 {len(df)} 条。")
    return df


def fetch_klines_between(start_ms: int, end_ms: int) -> pd.DataFrame:
    """按毫秒时间戳区间精确拉取真实联网 K 线。"""
    validate_ms_timestamp(start_ms, "start_ms")
    validate_ms_timestamp(end_ms, "end_ms")

    if end_ms < start_ms:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    failures = []
    sources = [
        ("Binance", lambda: _fetch_binance_like_between("Binance", BINANCE_BASE_URL, start_ms, end_ms)),
        (
            "Binance Vision",
            lambda: _fetch_binance_like_between(
                "Binance Vision",
                BINANCE_VISION_BASE_URL,
                start_ms,
                end_ms,
            ),
        ),
        ("OKX", lambda: _fetch_okx_between(start_ms, end_ms)),
    ]

    for source_name, fetcher in sources:
        try:
            df = fetcher()
            if not df.empty:
                return df
            failures.append(f"{source_name}: empty")
        except Exception as exc:
            failures.append(f"{source_name}: {type(exc).__name__}: {exc}")
            print(f"[数据源] {failures[-1]}")

    raise RuntimeError("所有真实联网 K 线数据源均不可用：" + " | ".join(failures))


def fetch_recent_klines(minutes: int) -> pd.DataFrame:
    """
    拉取最近指定分钟数的 1m K 线。

    注意：
    - 此函数始终从网络拉取；
    - 回测请优先使用 get_recent_klines_with_cache()。
    """
    now_ms = utc_ms_now()
    start_ms = now_ms - int(minutes) * INTERVAL_MS

    df = fetch_klines_between(start_ms=start_ms, end_ms=now_ms)

    if df.empty:
        return df

    now_ms = utc_ms_now()

    # 剔除尚未收盘 K 线
    df = df[df["close_time"] < now_ms].copy()
    df = clean_kline_dataframe(df)

    return df


def get_recent_klines_with_cache(
    minutes: int,
    path=HISTORY_CSV,
    update_if_needed: bool = True,
) -> pd.DataFrame:
    """
    本地缓存优先获取最近 N 分钟 K 线。

    逻辑：
    1. 先读取 data/BTCUSDT_1m_history.csv；
    2. 如果本地数据覆盖所需区间，直接返回；
    3. 如果没有本地数据，或数据不足，则只下载缺失区间；
    4. 下载后与本地数据合并保存；
    5. 返回最近 N 分钟数据。

    这样回测不会每次都重新从币安下载。
    """
    history_df = load_history_csv(path)
    now_ms = utc_ms_now()
    if not update_if_needed and not history_df.empty:
        cache_end_ms = int(history_df["timestamp"].max()) + INTERVAL_MS
        if cache_end_ms > now_ms:
            now_ms = cache_end_ms
    required_start_ms = now_ms - int(minutes) * INTERVAL_MS

    need_download = False
    download_start_ms = required_start_ms

    if history_df.empty:
        print(f"[数据缓存] 未发现本地历史数据，将下载最近 {minutes} 分钟数据。")
        need_download = True
        download_start_ms = required_start_ms
    else:
        min_ts = int(history_df["timestamp"].min())
        max_ts = int(history_df["timestamp"].max())

        print("[数据缓存] 已发现本地历史数据：")
        print(f"  文件: {path}")
        print(f"  本地起点: {ms_to_beijing_time(min_ts)}")
        print(f"  本地终点: {ms_to_beijing_time(max_ts)}")
        print(f"  本地条数: {len(history_df)}")

        if min_ts > required_start_ms:
            print("[数据缓存] 本地历史起点晚于回测所需起点，需要补充更早数据。")
            need_download = True
            download_start_ms = required_start_ms

        elif update_if_needed and max_ts < now_ms - 2 * INTERVAL_MS:
            print("[数据缓存] 本地历史不是最新，需要补充最新数据。")
            need_download = True
            download_start_ms = max_ts + INTERVAL_MS

        else:
            print("[数据缓存] 本地历史数据覆盖所需区间，直接使用本地数据。")

    if need_download:
        try:
            print("[数据缓存] 开始下载缺失 K 线：")
            print(f"  下载起点: {ms_to_beijing_time(download_start_ms)}")
            print(f"  下载终点: {ms_to_beijing_time(now_ms)}")

            new_df = fetch_klines_between(
                start_ms=download_start_ms,
                end_ms=now_ms,
            )

            if not new_df.empty:
                history_df = merge_history(history_df, new_df)
                save_history_csv(history_df, path)
                print(f"[数据缓存] 下载并合并完成，本地总条数: {len(history_df)}")
            else:
                print("[数据缓存] 下载结果为空，将尝试使用已有本地数据。")

        except Exception as exc:
            print(f"[数据缓存] 下载失败: {type(exc).__name__}: {exc}")
            print("[数据缓存] 将尝试使用已有本地数据继续。")

    history_df = clean_kline_dataframe(history_df)

    if history_df.empty:
        raise RuntimeError("本地无历史数据，且下载失败，无法继续。")

    # 剔除尚未收盘 K 线
    current_ms = utc_ms_now()
    if not update_if_needed:
        current_ms = max(current_ms, now_ms)
    history_df = history_df[history_df["close_time"] < current_ms].copy()
    history_df = clean_kline_dataframe(history_df)

    result = history_df[history_df["timestamp"] >= required_start_ms].copy()
    result = clean_kline_dataframe(result)

    if result.empty:
        raise RuntimeError("本地历史数据未覆盖所需回测区间。")

    print("[数据缓存] 本次返回数据：")
    print(f"  起点: {ms_to_beijing_time(int(result['timestamp'].min()))}")
    print(f"  终点: {ms_to_beijing_time(int(result['timestamp'].max()))}")
    print(f"  条数: {len(result)}")

    return result


def fetch_latest_closed_kline() -> pd.DataFrame:
    """获取最新已收盘的 1 条 K 线。"""
    df = fetch_recent_klines(minutes=5)

    if df.empty:
        return df

    latest = df.loc[[df["timestamp"].idxmax()]].copy()
    return latest.reset_index(drop=True)


def get_price_at_timestamp(df: pd.DataFrame, target_ms: int):
    """通过 13 位毫秒时间戳精确匹配价格。"""
    validate_ms_timestamp(target_ms, "target_ms")

    if df.empty:
        return None

    matched = df[df["timestamp"] == target_ms]

    if matched.empty:
        return None

    return float(matched.iloc[0]["close"])
