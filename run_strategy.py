# run_strategy.py

import time

import numpy as np
import pandas as pd

from config import (
    BACKTEST_PROGRESS_EVERY,
    CSV_COLUMNS,
    LABEL_NEUTRAL_THRESHOLD,
    LONG_SIGNAL_THRESHOLD,
    MIN_SIGNALS_FOR_THRESHOLD_SEARCH,
    PREDICT_HORIZON_MINUTES,
    PROB_BIN_WIDTH,
    SHORT_SIGNAL_THRESHOLD,
    USE_LABEL_NEUTRAL_ZONE,
)
from data_download import ms_to_beijing_time
from features import build_features
from trainer import ProbabilityStabilityFilter, train_overfit_model, train_validation_model


def actual_direction(current_price: float, future_price: float) -> str:
    return "up" if future_price > current_price else "down"


def raw_probability_direction(p_up: float) -> str:
    """
    只根据概率阈值生成原始方向信号。

    该信号不经过稳定性过滤、regime 过滤和质量门控，主要用于严格回测后分析
    “模型原始概率是否具有可交易方向性”。正式信号仍然使用 predicted_direction。
    """
    if p_up >= LONG_SIGNAL_THRESHOLD:
        return "up"

    if p_up <= SHORT_SIGNAL_THRESHOLD:
        return "down"

    return "no_trade"


def high_confidence_report(result_df: pd.DataFrame) -> dict:
    """
    只统计真正发出 up/down 信号的样本。
    no_trade 不参与胜率统计。
    """
    base = {
        "total_rows": 0,
        "valid_signals": 0,
        "valid_signal_ratio": None,
        "valid_win_rate": None,
        "long_signals": 0,
        "long_win_rate": None,
        "short_signals": 0,
        "short_win_rate": None,
        "no_trade_rows": 0,
    }

    if result_df.empty:
        return base

    base["total_rows"] = len(result_df)

    signal_df = result_df[result_df["is_valid_signal"] == True].copy()
    no_trade_df = result_df[result_df["is_valid_signal"] == False].copy()

    base["no_trade_rows"] = len(no_trade_df)

    if signal_df.empty:
        base["valid_signal_ratio"] = 0.0
        return base

    base["valid_signals"] = len(signal_df)
    base["valid_signal_ratio"] = float(len(signal_df) / len(result_df))
    base["valid_win_rate"] = float(signal_df["is_correct"].mean())

    long_df = signal_df[signal_df["predicted_direction"] == "up"].copy()
    short_df = signal_df[signal_df["predicted_direction"] == "down"].copy()

    base["long_signals"] = len(long_df)
    base["short_signals"] = len(short_df)

    if not long_df.empty:
        base["long_win_rate"] = float(long_df["is_correct"].mean())

    if not short_df.empty:
        base["short_win_rate"] = float(short_df["is_correct"].mean())

    return base


def raw_signal_report(result_df: pd.DataFrame) -> dict:
    """
    统计只使用概率阈值时的原始信号表现。
    """
    base = {
        "total_rows": 0,
        "raw_valid_signals": 0,
        "raw_valid_signal_ratio": None,
        "raw_win_rate": None,
        "raw_long_signals": 0,
        "raw_long_win_rate": None,
        "raw_short_signals": 0,
        "raw_short_win_rate": None,
        "raw_no_trade_rows": 0,
    }

    if result_df.empty or "raw_predicted_direction" not in result_df.columns:
        return base

    base["total_rows"] = len(result_df)
    signal_df = result_df[result_df["raw_predicted_direction"].isin(["up", "down"])].copy()
    base["raw_valid_signals"] = len(signal_df)
    base["raw_no_trade_rows"] = len(result_df) - len(signal_df)

    if signal_df.empty:
        base["raw_valid_signal_ratio"] = 0.0
        return base

    base["raw_valid_signal_ratio"] = float(len(signal_df) / len(result_df))
    base["raw_win_rate"] = float(signal_df["raw_is_correct"].mean())

    long_df = signal_df[signal_df["raw_predicted_direction"] == "up"].copy()
    short_df = signal_df[signal_df["raw_predicted_direction"] == "down"].copy()

    base["raw_long_signals"] = len(long_df)
    base["raw_short_signals"] = len(short_df)

    if not long_df.empty:
        base["raw_long_win_rate"] = float(long_df["raw_is_correct"].mean())

    if not short_df.empty:
        base["raw_short_win_rate"] = float(short_df["raw_is_correct"].mean())

    return base


def probability_bin_report(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 p_up 分桶统计真实上涨比例。

    用来判断模型概率是否有区分度。
    """
    if result_df.empty:
        return pd.DataFrame()

    df = result_df.copy()

    bins = np.arange(0.0, 1.0 + PROB_BIN_WIDTH, PROB_BIN_WIDTH)
    labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]

    df["prob_bin"] = pd.cut(
        df["up_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    rows = []

    for bin_name, g in df.groupby("prob_bin", observed=False):
        if g.empty:
            continue

        actual_up_rate = (g["actual_direction"] == "up").mean()

        rows.append(
            {
                "prob_bin": str(bin_name),
                "count": len(g),
                "actual_up_rate": float(actual_up_rate),
                "avg_p_up": float(g["up_probability"].mean()),
            }
        )

    return pd.DataFrame(rows)


def threshold_search_report(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    自动搜索不同 long_threshold / short_threshold 下的信号胜率。

    目的：
    找到在信号数足够的前提下，哪些阈值组合更合理。
    """
    if result_df.empty:
        return pd.DataFrame()

    rows = []

    long_thresholds = np.arange(0.50, 0.91, 0.05)
    short_thresholds = np.arange(0.50, 0.09, -0.05)

    for long_th in long_thresholds:
        long_df = result_df[result_df["up_probability"] >= long_th].copy()

        if not long_df.empty:
            long_win_rate = (long_df["actual_direction"] == "up").mean()
        else:
            long_win_rate = None

        rows.append(
            {
                "side": "long",
                "threshold": round(float(long_th), 2),
                "signals": len(long_df),
                "win_rate": None if long_win_rate is None else float(long_win_rate),
            }
        )

    for short_th in short_thresholds:
        short_df = result_df[result_df["up_probability"] <= short_th].copy()

        if not short_df.empty:
            short_win_rate = (short_df["actual_direction"] == "down").mean()
        else:
            short_win_rate = None

        rows.append(
            {
                "side": "short",
                "threshold": round(float(short_th), 2),
                "signals": len(short_df),
                "win_rate": None if short_win_rate is None else float(short_win_rate),
            }
        )

    report = pd.DataFrame(rows)

    if report.empty:
        return report

    enough = report[report["signals"] >= MIN_SIGNALS_FOR_THRESHOLD_SEARCH].copy()

    if enough.empty:
        return report.sort_values(["side", "signals"], ascending=[True, False])

    return enough.sort_values(
        ["win_rate", "signals"],
        ascending=[False, False],
    )


def strict_walk_forward_backtest(
    df: pd.DataFrame,
    train_window_minutes: int,
    step_minutes: int,
    model_update_minutes: int,
    min_train_samples: int,
    max_steps: int,
    progress_every: int = BACKTEST_PROGRESS_EVERY,
) -> pd.DataFrame:
    """
    严格时序滚动验证回测。

    - 模型按 model_update_minutes 滚动更新；
    - 每次更新只使用更新时间点之前的数据训练；
    - 更新间隔内的预测使用最近一次训练好的模型；
    - 当前点之后的数据只用于事后验证；
    - 固定目标：future_close[t+10] > current_close[t]；
    - no_trade 不参与胜率统计。
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    close_by_timestamp = df.set_index("timestamp")["close"]
    feature_df = build_features(df)

    n = len(df)
    horizon = PREDICT_HORIZON_MINUTES

    if n <= train_window_minutes + horizon + 1:
        print(
            "[严格回测] 数据量不足："
            f"n={n}, train_window={train_window_minutes}, horizon={horizon}"
        )
        return pd.DataFrame()

    if step_minutes < 1:
        raise ValueError("step_minutes 必须 >= 1。")

    if model_update_minutes < 1:
        raise ValueError("model_update_minutes 必须 >= 1。")

    start_idx = train_window_minutes
    end_idx = n - horizon - 1

    candidate_indices = list(range(start_idx, end_idx, step_minutes))

    if max_steps is not None:
        candidate_indices = candidate_indices[-max_steps:]

    total_steps = len(candidate_indices)

    print("[严格回测] 参数：")
    print(f"  数据总量: {n}")
    print(f"  训练窗口: {train_window_minutes} 分钟")
    print(f"  预测周期: {horizon} 分钟")
    print(f"  步长: {step_minutes} 分钟")
    print(f"  模型更新间隔: {model_update_minutes} 分钟")
    print(f"  最大步数: {max_steps}")
    print(f"  实际回测步数: {total_steps}")
    print(f"  做多阈值: p_up >= {LONG_SIGNAL_THRESHOLD}")
    print(f"  做空阈值: p_up <= {SHORT_SIGNAL_THRESHOLD}")
    print(f"  是否使用训练标签灰区: {USE_LABEL_NEUTRAL_ZONE}")
    if USE_LABEL_NEUTRAL_ZONE:
        print(f"  标签灰区阈值: ±{LABEL_NEUTRAL_THRESHOLD}")
    print("  说明: 预测按步长推进，模型只在更新间隔到达时用最近历史重新训练。")

    results = []
    started = time.time()
    model = None
    model_trained_at_time = None
    next_model_update_idx = None
    train_count = 0
    signal_filter = ProbabilityStabilityFilter()

    for step_no, i in enumerate(candidate_indices, start=1):
        point_time = ms_to_beijing_time(int(df.iloc[i]["timestamp"]))
        print(f"[严格回测] {step_no}/{total_steps} 开始：{point_time}")

        needs_model_update = (
            model is None
            or next_model_update_idx is None
            or i >= next_model_update_idx
        )

        if needs_model_update:
            train_start = max(0, i - train_window_minutes)
            train_df = df.iloc[train_start:i].copy()

            if len(train_df) < min_train_samples:
                print(
                    f"[严格回测] 跳过：训练样本不足 "
                    f"{len(train_df)} < {min_train_samples}"
                )
                continue

            try:
                model = train_validation_model(train_df)
                model_trained_at_time = point_time
                next_model_update_idx = i + model_update_minutes
                train_count += 1
                print(
                    f"[严格回测] 模型已更新："
                    f"train_rows={len(train_df)}, "
                    f"trained_at={model_trained_at_time}, "
                    f"next_update_index>={next_model_update_idx}"
                )
            except Exception as exc:
                print(f"[严格回测] 训练失败 step={step_no}, index={i}: {exc}")
                continue

        try:
            feat_cols = model.feature_cols
            latest = feature_df.iloc[[i]].copy()

            if latest[feat_cols].isna().any(axis=None):
                print("[严格回测] 跳过：当前点特征存在 NaN。")
                continue

            pred = model.predict_one(latest[feat_cols], signal_filter=signal_filter)

            current_row = df.iloc[i]
            current_price = float(current_row["close"])
            future_ms = int(current_row["timestamp"]) + horizon * 60_000

            if future_ms not in close_by_timestamp.index:
                print(
                    "[严格回测] 跳过：无法按 13 位毫秒时间戳匹配未来价格 "
                    f"{ms_to_beijing_time(future_ms)}"
                )
                continue

            future_price = float(close_by_timestamp.loc[future_ms])
            future_return = future_price / current_price - 1

            pred_dir = pred["predicted_direction"]
            act_dir = actual_direction(current_price, future_price)
            raw_pred_dir = raw_probability_direction(float(pred["up_probability"]))

            if pred_dir == "no_trade":
                correct = False
            else:
                correct = pred_dir == act_dir

            if raw_pred_dir == "no_trade":
                raw_correct = False
            else:
                raw_correct = raw_pred_dir == act_dir

            row = {
                "timestamp": ms_to_beijing_time(int(current_row["timestamp"])),
                "current_price": current_price,
                "future_price": future_price,
                "future_return": future_return,
                "predicted_direction": pred_dir,
                "actual_direction": act_dir,
                "raw_predicted_direction": raw_pred_dir,
                "up_probability": pred["up_probability"],
                "confidence": pred["confidence"],
                "is_valid_signal": pred["is_valid_signal"],
                "is_correct": correct,
                "raw_is_correct": raw_correct,
                "model_trained_at": model_trained_at_time,
            }

            results.append(row)

            print(
                f"[严格回测] 完成："
                f"p_up={pred['up_probability']:.4f}, "
                f"raw_signal={raw_pred_dir}, "
                f"signal={pred_dir}, "
                f"actual={act_dir}, "
                f"future_return={future_return:.6f}, "
                f"model_at={model_trained_at_time}, "
                f"valid={pred['is_valid_signal']}, "
                f"correct={correct}"
            )

        except Exception as exc:
            print(f"[严格回测] 预测失败 step={step_no}, index={i}: {exc}")
            continue

        if step_no % progress_every == 0 or step_no == total_steps:
            elapsed = time.time() - started
            result_df = pd.DataFrame(results)
            report = high_confidence_report(result_df)
            raw_report = raw_signal_report(result_df)

            print("[严格回测] 当前进度报告：")
            print(f"  已完成: {step_no}/{total_steps}")
            print(f"  已耗时: {elapsed:.1f} 秒")
            print(f"  模型更新次数: {train_count}")
            print(f"  总预测数: {report['total_rows']}")
            print(f"  有效信号数: {report['valid_signals']}")
            print(f"  no_trade 数: {report['no_trade_rows']}")
            print(f"  有效信号占比: {report['valid_signal_ratio']}")
            print(f"  有效信号胜率: {report['valid_win_rate']}")
            print(f"  做多信号数: {report['long_signals']}")
            print(f"  做多胜率: {report['long_win_rate']}")
            print(f"  做空信号数: {report['short_signals']}")
            print(f"  做空胜率: {report['short_win_rate']}")
            print("[严格回测] 原始概率信号进度报告：")
            print(f"  原始有效信号数: {raw_report['raw_valid_signals']}")
            print(f"  原始 no_trade 数: {raw_report['raw_no_trade_rows']}")
            print(f"  原始有效信号占比: {raw_report['raw_valid_signal_ratio']}")
            print(f"  原始有效信号胜率: {raw_report['raw_win_rate']}")
            print(f"  原始做多信号数: {raw_report['raw_long_signals']}")
            print(f"  原始做多胜率: {raw_report['raw_long_win_rate']}")
            print(f"  原始做空信号数: {raw_report['raw_short_signals']}")
            print(f"  原始做空胜率: {raw_report['raw_short_win_rate']}")

    result_df = pd.DataFrame(results)

    print("[严格回测] 全部完成。")
    print(high_confidence_report(result_df))
    print("[严格回测] 原始概率信号汇总。")
    print(raw_signal_report(result_df))

    return result_df


def leaked_training_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    第一阶段：模型训练回测。

    该函数故意使用全量数据训练并在同一批数据上预测，允许未来信息参与训练。
    结果只用于观察局部拟合效果，禁止写入 predictions.csv，也不代表实盘表现。
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)

    model = train_overfit_model(df)
    feat_df = build_features(df)
    feat_cols = model.feature_cols
    feat_df = feat_df.dropna(subset=feat_cols).copy()

    close_by_timestamp = df.set_index("timestamp")["close"]
    results = []
    signal_filter = ProbabilityStabilityFilter()

    for _, row in feat_df.iterrows():
        timestamp_ms = int(row["timestamp"])
        future_ms = timestamp_ms + PREDICT_HORIZON_MINUTES * 60_000

        if future_ms not in close_by_timestamp.index:
            continue

        current_price = float(row["close"])
        future_price = float(close_by_timestamp.loc[future_ms])
        pred = model.predict_one(row[feat_cols].to_frame().T, signal_filter=signal_filter)
        pred_dir = pred["predicted_direction"]
        act_dir = actual_direction(current_price, future_price)
        correct = False if pred_dir == "no_trade" else pred_dir == act_dir

        results.append(
            {
                "timestamp": ms_to_beijing_time(timestamp_ms),
                "current_price": current_price,
                "future_price": future_price,
                "predicted_direction": pred_dir,
                "actual_direction": act_dir,
                "up_probability": pred["up_probability"],
                "confidence": pred["confidence"],
                "is_valid_signal": pred["is_valid_signal"],
                "is_correct": correct,
            }
        )

    return pd.DataFrame(results, columns=CSV_COLUMNS)
