# realtime.py

import json
import time

import pandas as pd

from config import (
    CSV_COLUMNS,
    PENDING_FILE,
    PREDICTIONS_CSV,
    PREDICT_HORIZON_MS,
    REALTIME_INTERVAL_SECONDS,
    RETRAIN_INTERVAL_SECONDS,
    TRAIN_MINUTES,
)
from data_download import (
    fetch_klines_between,
    fetch_recent_klines,
    get_price_at_timestamp,
    ms_to_beijing_time,
)
from features import make_realtime_features
from run_strategy import actual_direction
from trainer import ProbabilityStabilityFilter, load_model, train_and_save


def ensure_predictions_csv():
    """
    确保 predictions.csv 存在且字段顺序正确。
    """
    PREDICTIONS_CSV.parent.mkdir(exist_ok=True)

    if not PREDICTIONS_CSV.exists():
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
        print(f"[CSV] 已创建: {PREDICTIONS_CSV}")
        return

    try:
        df_head = pd.read_csv(
            PREDICTIONS_CSV,
            nrows=1,
            dtype=object,
            keep_default_na=False,
        )

        if list(df_head.columns) != CSV_COLUMNS:
            raise ValueError(
                "predictions.csv 字段顺序不符合要求。"
                f"\n当前字段: {list(df_head.columns)}"
                f"\n要求字段: {CSV_COLUMNS}"
            )

    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
        print(f"[CSV] 空文件已重建: {PREDICTIONS_CSV}")


def read_predictions_csv() -> pd.DataFrame:
    """
    读取 predictions.csv，全部按 object 处理，避免 dtype 回填错误。
    """
    ensure_predictions_csv()

    try:
        df = pd.read_csv(
            PREDICTIONS_CSV,
            dtype=object,
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[CSV_COLUMNS].copy()

    for col in CSV_COLUMNS:
        df[col] = df[col].astype(object)

    return df


def write_predictions_csv(df: pd.DataFrame):
    """
    整体写回 predictions.csv。
    """
    df = df.copy()

    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[CSV_COLUMNS].fillna("").copy()

    for col in CSV_COLUMNS:
        df[col] = df[col].astype(object)

    df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")


def append_prediction_row_immediately(row: dict):
    """
    预测后立即写入 predictions.csv。
    future_price、actual_direction、is_correct 暂时为空。
    """
    df = read_predictions_csv()

    timestamp = str(row["timestamp"])

    if not df.empty and (df["timestamp"].astype(str) == timestamp).any():
        print(f"[CSV] 当前 timestamp 已存在，跳过重复写入: {timestamp}")
        return

    row_full = {col: row.get(col, "") for col in CSV_COLUMNS}
    row_full["timestamp"] = timestamp

    df_new = pd.DataFrame([row_full], columns=CSV_COLUMNS)

    for col in CSV_COLUMNS:
        df_new[col] = df_new[col].astype(object)

    df = pd.concat([df, df_new], ignore_index=True)
    write_predictions_csv(df)

    print(f"[CSV] 预测已立即写入: {timestamp}")


def update_prediction_row(
    timestamp: str,
    future_price: float,
    actual_dir: str,
    is_correct,
    pending_item: dict | None = None,
):
    """
    根据 timestamp 回填 future_price、actual_direction、is_correct。

    如果找不到原始行，自动根据 pending_item 补建。
    """
    df = read_predictions_csv()

    timestamp = str(timestamp)
    mask = df["timestamp"].astype(str) == timestamp

    if not mask.any():
        if pending_item is None:
            print(f"[CSV] 未找到待回填行，且无 pending_item，无法补建: {timestamp}")
            return False

        print(f"[CSV] 未找到待回填行，将根据 pending 自动补建: {timestamp}")

        recovered_row = {
            "timestamp": timestamp,
            "current_price": pending_item.get("current_price", ""),
            "future_price": "",
            "predicted_direction": pending_item.get("predicted_direction", ""),
            "actual_direction": "",
            "up_probability": pending_item.get("up_probability", ""),
            "confidence": pending_item.get("confidence", ""),
            "is_valid_signal": pending_item.get("is_valid_signal", ""),
            "is_correct": "",
        }

        df_new = pd.DataFrame([recovered_row], columns=CSV_COLUMNS)

        for col in CSV_COLUMNS:
            df_new[col] = df_new[col].astype(object)

        df = pd.concat([df, df_new], ignore_index=True)
        mask = df["timestamp"].astype(str) == timestamp

    for col in ["future_price", "actual_direction", "is_correct"]:
        df[col] = df[col].astype(object)

    df.loc[mask, "future_price"] = str(float(future_price))
    df.loc[mask, "actual_direction"] = str(actual_dir)
    df.loc[mask, "is_correct"] = "" if is_correct == "" else str(bool(is_correct))

    write_predictions_csv(df)

    print(
        f"[CSV] 已回填: {timestamp}, "
        f"future_price={future_price}, "
        f"actual_direction={actual_dir}, "
        f"is_correct={is_correct}"
    )

    return True


def load_pending_predictions() -> list:
    if not PENDING_FILE.exists():
        return []

    items = []

    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print("[pending] 跳过损坏行。")

    return items


def save_pending_predictions(items: list):
    PENDING_FILE.parent.mkdir(exist_ok=True)

    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def add_pending_prediction(item: dict):
    items = load_pending_predictions()

    existing_ts = {int(x["timestamp_ms"]) for x in items}
    current_ts = int(item["timestamp_ms"])

    if current_ts not in existing_ts:
        items.append(item)
        save_pending_predictions(items)

        print(
            "[pending] 新增待验证预测:",
            ms_to_beijing_time(current_ts),
            f"当前 pending 数量={len(items)}",
        )
    else:
        print(
            "[pending] 当前 K 线已存在待验证记录，跳过重复写入:",
            ms_to_beijing_time(current_ts),
        )


def load_recent_klines_for_runtime(minutes: int) -> pd.DataFrame:
    """
    实时运行数据入口。

    只使用真实联网数据源。下载层会按 Binance -> Binance Vision -> OKX 顺序兜底。
    如果所有真实联网源均失败，本轮训练或预测直接失败，不使用本地历史 CSV。
    """
    return fetch_recent_klines(minutes)


def train_model_now():
    """
    拉取最近 48 小时数据并训练实时模型。
    """
    print("[训练] 拉取最近 48 小时 1m K 线...")
    df = load_recent_klines_for_runtime(TRAIN_MINUTES)

    if df.empty:
        raise RuntimeError("训练数据为空。")

    print(f"[训练] 数据量: {len(df)}")

    model = train_and_save(df, mode="overfit")

    print("[训练] 单方向模型训练完成并已保存。")
    return model


def make_live_prediction(model, signal_filter: ProbabilityStabilityFilter):
    """
    生成实时预测。

    预测后立即写 CSV；10 分钟后回填。
    """
    print("[实时] 拉取最近 48 小时 K 线用于构造最新特征...")
    df = load_recent_klines_for_runtime(TRAIN_MINUTES)

    if df.empty:
        print("[实时] 未获取到 K 线数据。")
        return

    X_latest, latest_row = make_realtime_features(df, model.feature_cols)

    if X_latest is None:
        print("[实时] 特征不足，跳过本轮预测。")
        return

    timestamp_ms = int(latest_row.iloc[0]["timestamp"])
    timestamp_str = ms_to_beijing_time(timestamp_ms)
    current_price = float(latest_row.iloc[0]["close"])

    pred = model.predict_one(X_latest, signal_filter=signal_filter)

    row = {
        "timestamp": timestamp_str,
        "current_price": current_price,
        "future_price": "",
        "predicted_direction": pred["predicted_direction"],
        "actual_direction": "",
        "up_probability": pred["up_probability"],
        "confidence": pred["confidence"],
        "is_valid_signal": pred["is_valid_signal"],
        "is_correct": "",
    }

    append_prediction_row_immediately(row)

    item = {
        "timestamp_ms": timestamp_ms,
        "timestamp": timestamp_str,
        "current_price": current_price,
        "predicted_direction": pred["predicted_direction"],
        "up_probability": pred["up_probability"],
        "confidence": pred["confidence"],
        "is_valid_signal": pred["is_valid_signal"],
    }

    add_pending_prediction(item)

    print(
        "[实时预测]",
        timestamp_str,
        f"price={current_price:.2f}",
        f"p_up={pred['up_probability']:.4f}",
        f"signal={pred['predicted_direction']}",
        f"confidence={pred['confidence']:.4f}",
        f"valid={pred['is_valid_signal']}",
        "已立即写入 CSV，等待 10 分钟后回填。",
    )


def validate_pending_predictions():
    """
    验证已经到期的 pending 预测，并回填 predictions.csv。
    """
    items = load_pending_predictions()

    if not items:
        print("[验证] 当前没有 pending 预测。")
        return

    now_ms = int(time.time() * 1000)

    print(f"[验证] 当前 pending 数量: {len(items)}")

    remaining = []
    completed_count = 0

    items = sorted(items, key=lambda x: int(x["timestamp_ms"]))

    for item in items:
        timestamp_ms = int(item["timestamp_ms"])
        timestamp_str = item.get("timestamp", ms_to_beijing_time(timestamp_ms))
        future_ms = timestamp_ms + PREDICT_HORIZON_MS

        if now_ms < future_ms:
            remaining.append(item)

            wait_seconds = int((future_ms - now_ms) / 1000)

            print(
                "[验证] 尚未到验证时间:",
                timestamp_str,
                f"还需等待约 {wait_seconds} 秒",
            )
            continue

        print(
            "[验证] 开始回填:",
            timestamp_str,
            "→",
            ms_to_beijing_time(future_ms),
        )

        try:
            future_df = fetch_klines_between(
                start_ms=future_ms,
                end_ms=future_ms + 60_000,
            )

            future_price = get_price_at_timestamp(future_df, future_ms)

            if future_price is None:
                print(
                    "[验证] 未匹配到 future_price，保留 pending:",
                    ms_to_beijing_time(future_ms),
                )
                remaining.append(item)
                continue

            current_price = float(item["current_price"])
            pred_dir = item["predicted_direction"]
            act_dir = actual_direction(current_price, float(future_price))

            if pred_dir == "no_trade":
                correct = ""
            else:
                correct = pred_dir == act_dir

            updated = update_prediction_row(
                timestamp=timestamp_str,
                future_price=float(future_price),
                actual_dir=act_dir,
                is_correct=correct,
                pending_item=item,
            )

            if updated:
                completed_count += 1

                print(
                    "[验证完成]",
                    timestamp_str,
                    f"current={current_price:.2f}",
                    f"future={float(future_price):.2f}",
                    f"pred={pred_dir}",
                    f"actual={act_dir}",
                    f"correct={correct}",
                )
            else:
                print("[验证] CSV 回填失败，继续保留 pending:", timestamp_str)
                remaining.append(item)

        except Exception as exc:
            print(
                "[验证] 回填失败，保留 pending:",
                timestamp_str,
                f"{type(exc).__name__}: {exc}",
            )
            remaining.append(item)

    save_pending_predictions(remaining)

    print(
        f"[验证] 本轮完成回填 {completed_count} 条，"
        f"剩余 pending {len(remaining)} 条。"
    )


def realtime_loop():
    """
    实时运行主循环。

    - 每 60 秒预测一次；
    - 每 30 分钟重训一次；
    - 预测后立即写 CSV；
    - 10 分钟后回填。
    """
    ensure_predictions_csv()

    print("[启动] predictions.csv 路径:", PREDICTIONS_CSV)
    print("[启动] pending 文件路径:", PENDING_FILE)

    model = load_model()
    last_train_time = 0
    signal_filter = ProbabilityStabilityFilter()

    if model is None:
        print("[启动] 未发现模型，开始首次训练。")
        model = train_model_now()
        last_train_time = time.time()
    else:
        print("[启动] 已加载本地模型。")
        last_train_time = time.time()

    while True:
        loop_start = time.time()

        print("=" * 80)
        print("[循环] 新一轮 realtime 任务开始。")

        try:
            now = time.time()

            if now - last_train_time >= RETRAIN_INTERVAL_SECONDS:
                print("[定时训练] 到达 30 分钟重训周期。")
                model = train_model_now()
                last_train_time = now

            validate_pending_predictions()
            make_live_prediction(model, signal_filter)

        except Exception as exc:
            print(f"[错误] {type(exc).__name__}: {exc}")

        elapsed = time.time() - loop_start
        sleep_seconds = max(1, REALTIME_INTERVAL_SECONDS - elapsed)

        print(f"[循环] 本轮耗时 {elapsed:.1f} 秒，休眠 {sleep_seconds:.1f} 秒。")
        time.sleep(sleep_seconds)
