# realtime.py

import json
import time

import pandas as pd
import requests

from config import (
    CSV_COLUMNS,
    ENABLE_WECHAT_NOTIFICATIONS,
    PENDING_FILE,
    PREDICTIONS_CSV,
    PREDICT_HORIZON_MS,
    PREDICT_HORIZON_MINUTES,
    REALTIME_INTERVAL_SECONDS,
    REQUEST_VERIFY_SSL,
    RETRAIN_INTERVAL_SECONDS,
    TRAIN_MINUTES,
    WECHAT_REQUEST_TIMEOUT,
    WECHAT_WEBHOOK_URL,
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


def direction_to_cn(direction: str) -> str:
    if direction == "up":
        return "上涨"
    if direction == "down":
        return "下跌"
    return "无交易"


def format_price(price: float) -> str:
    return f"${float(price):,.2f}"


def format_percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def is_truthy(value) -> bool:
    return str(value).lower() == "true"


def send_wechat_markdown(content: str) -> bool:
    """
    发送企业微信 markdown 通知。

    通知失败不影响预测和 CSV 写入。
    """
    if not ENABLE_WECHAT_NOTIFICATIONS:
        return False

    if not WECHAT_WEBHOOK_URL:
        print("[通知] 未配置企业微信 webhook，跳过。")
        return False

    payload = {
        "msgtype": "markdown",
        "markdown": {
            "content": content,
        },
    }

    try:
        resp = requests.post(
            WECHAT_WEBHOOK_URL,
            json=payload,
            timeout=WECHAT_REQUEST_TIMEOUT,
            verify=REQUEST_VERIFY_SSL,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("errcode") != 0:
            print(f"[通知] 企业微信返回异常: {data}")
            return False

        print("[通知] 企业微信通知已发送。")
        return True

    except Exception as exc:
        print(f"[通知] 企业微信通知失败: {type(exc).__name__}: {exc}")
        return False


def high_confidence_stats() -> tuple[int, int, float | None]:
    """
    统计已完成验证的高置信信号命中情况。
    """
    df = read_predictions_csv()

    if df.empty:
        return 0, 0, None

    valid_mask = df["is_valid_signal"].astype(str).str.lower() == "true"
    done_mask = df["is_correct"].astype(str).str.lower().isin({"true", "false"})
    signal_df = df[valid_mask & done_mask].copy()

    total = len(signal_df)
    if total == 0:
        return 0, 0, None

    correct = int((signal_df["is_correct"].astype(str).str.lower() == "true").sum())
    return correct, total, correct / total


def notify_high_confidence_signal(row: dict):
    if not is_truthy(row.get("is_valid_signal")):
        return

    pred_dir = row.get("predicted_direction")
    if pred_dir not in {"up", "down"}:
        return

    direction_icon = "📈" if pred_dir == "up" else "📉"
    content = "\n".join(
        [
            f"{direction_icon} 方向：{direction_to_cn(pred_dir)}",
            f"⏰ 预测时间: {row['timestamp']}",
            f"💰 当前价格：{format_price(float(row['current_price']))}",
            f"⭐ 置信度: {format_percent(float(row['confidence']))}",
            "🛡️ 风控：通过",
        ]
    )
    send_wechat_markdown(content)


def notify_signal_validation(
    *,
    timestamp: str,
    future_timestamp: str,
    current_price: float,
    future_price: float,
    predicted_direction: str,
    actual_direction_value: str,
    confidence: float,
    is_correct: bool,
):
    if predicted_direction not in {"up", "down"}:
        return

    correct, total, win_rate = high_confidence_stats()
    if win_rate is None:
        stats_line = "📊 高置信度预测中：暂无已完成验证记录。"
    else:
        stats_line = (
            f"📊 高置信度预测中：{correct}/{total}"
            f"（{win_rate * 100:.1f}%）预测正确！"
        )

    result_icon = "✅" if is_correct else "❌"
    note = "模型本次预测命中" if is_correct else "请注意，模型本次预测未命中"
    start_hm = timestamp[11:16]
    end_hm = future_timestamp[11:16]

    content = "\n".join(
        [
            f"{result_icon} {PREDICT_HORIZON_MINUTES}分钟比特币验证 | {start_hm} -> {end_hm}",
            f"⏰ 预测时间: {timestamp}",
            f"💰 预测时价格: {format_price(current_price)}",
            f"💰 验证时价格: {format_price(future_price)}",
            f"🔄 预测方向: {direction_to_cn(predicted_direction)}",
            f"🔄 实际方向: {direction_to_cn(actual_direction_value)}",
            f"⭐ 置信度: {format_percent(confidence)}",
            "",
            f"📌 {note}",
            stats_line,
        ]
    )
    send_wechat_markdown(content)


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
    notify_high_confidence_signal(row)

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
                future_timestamp_str = ms_to_beijing_time(future_ms)

                print(
                    "[验证完成]",
                    timestamp_str,
                    f"current={current_price:.2f}",
                    f"future={float(future_price):.2f}",
                    f"pred={pred_dir}",
                    f"actual={act_dir}",
                    f"correct={correct}",
                )

                if is_truthy(item.get("is_valid_signal")) and pred_dir in {"up", "down"}:
                    notify_signal_validation(
                        timestamp=timestamp_str,
                        future_timestamp=future_timestamp_str,
                        current_price=current_price,
                        future_price=float(future_price),
                        predicted_direction=pred_dir,
                        actual_direction_value=act_dir,
                        confidence=float(item.get("confidence", 0.0)),
                        is_correct=bool(correct),
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
