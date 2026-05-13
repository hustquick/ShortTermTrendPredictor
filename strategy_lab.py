# strategy_lab.py

import time

import pandas as pd

from config import (
    BACKTEST_MODEL_UPDATE_MINUTES,
    BACKTEST_TRAIN_WINDOW_MINUTES,
    BACKTEST_MIN_TRAIN_SAMPLES,
    PREDICT_HORIZON_MINUTES,
)
from data_download import ms_to_beijing_time
from features import build_features
from trainer import train_validation_model
from strategies.rules import default_strategies


def _actual_direction(current_price: float, future_price: float) -> str:
    return "up" if future_price > current_price else "down"


def _summarize_strategy_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out = []

    for strategy, g in df.groupby("strategy"):
        signal_df = g[g["direction"].isin(["up", "down"])].copy()
        long_df = signal_df[signal_df["direction"] == "up"].copy()
        short_df = signal_df[signal_df["direction"] == "down"].copy()

        out.append(
            {
                "strategy": strategy,
                "total": len(g),
                "signals": len(signal_df),
                "signal_ratio": None if len(g) == 0 else len(signal_df) / len(g),
                "accuracy": None if signal_df.empty else float(signal_df["correct"].mean()),
                "long_signals": len(long_df),
                "long_accuracy": None if long_df.empty else float(long_df["correct"].mean()),
                "short_signals": len(short_df),
                "short_accuracy": None if short_df.empty else float(short_df["correct"].mean()),
                "no_trade": int((g["direction"] == "no_trade").sum()),
            }
        )

    return pd.DataFrame(out).sort_values(
        ["accuracy", "signals"],
        ascending=[False, False],
        na_position="last",
    )


def run_multi_strategy_backtest(
    df: pd.DataFrame,
    max_steps: int | None = 2000,
    progress_every: int = 100,
) -> pd.DataFrame:
    """多策略严格时序测试。

    目标固定为高置信方向准确率：
    - 不计算手续费；
    - 不计算收益率目标；
    - 不做仓位和资金曲线；
    - 只统计 up/down 信号是否预测正确。
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    close_by_timestamp = df.set_index("timestamp")["close"]
    feature_df = build_features(df)
    strategies = default_strategies()

    horizon = PREDICT_HORIZON_MINUTES
    start_idx = BACKTEST_TRAIN_WINDOW_MINUTES
    end_idx = len(df) - horizon - 1

    candidate_indices = list(range(start_idx, end_idx, 1))
    if max_steps is not None:
        candidate_indices = candidate_indices[-max_steps:]

    print("[策略实验] 参数：")
    print(f"  数据总量: {len(df)}")
    print(f"  实际测试步数: {len(candidate_indices)}")
    print(f"  预测周期: {horizon} 分钟")
    print(f"  策略数量: {len(strategies)}")
    print("  目标: 高置信方向准确率，不考虑手续费、收益率和仓位。")

    model = None
    next_model_update_idx = None
    model_trained_at_time = None
    train_count = 0
    rows = []
    started = time.time()

    for step_no, i in enumerate(candidate_indices, start=1):
        point_time = ms_to_beijing_time(int(df.iloc[i]["timestamp"]))

        if model is None or next_model_update_idx is None or i >= next_model_update_idx:
            train_start = max(0, i - BACKTEST_TRAIN_WINDOW_MINUTES)
            train_df = df.iloc[train_start:i].copy()
            if len(train_df) < BACKTEST_MIN_TRAIN_SAMPLES:
                continue
            model = train_validation_model(train_df)
            model_trained_at_time = point_time
            next_model_update_idx = i + BACKTEST_MODEL_UPDATE_MINUTES
            train_count += 1
            print(f"[策略实验] 模型更新：step={step_no}, trained_at={model_trained_at_time}")

        feat_cols = model.feature_cols
        latest = feature_df.iloc[[i]].copy()
        if latest[feat_cols].isna().any(axis=None):
            continue

        current_row = df.iloc[i]
        current_price = float(current_row["close"])
        future_ms = int(current_row["timestamp"]) + horizon * 60_000
        if future_ms not in close_by_timestamp.index:
            continue

        future_price = float(close_by_timestamp.loc[future_ms])
        actual = _actual_direction(current_price, future_price)
        prediction = model.predict_one(latest[feat_cols], signal_filter=None)
        feature_row = latest.iloc[0]

        for strategy in strategies:
            decision = strategy.decide(feature_row, prediction)
            correct = None
            if decision.direction in {"up", "down"}:
                correct = decision.direction == actual

            rows.append(
                {
                    "timestamp": point_time,
                    "strategy": strategy.name,
                    "direction": decision.direction,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "actual_direction": actual,
                    "correct": correct,
                    "up_signal_probability": prediction.get("up_signal_probability"),
                    "down_signal_probability": prediction.get("down_signal_probability"),
                    "direction_edge": prediction.get("direction_edge"),
                    "model_trained_at": model_trained_at_time,
                }
            )

        if step_no % progress_every == 0 or step_no == len(candidate_indices):
            elapsed = time.time() - started
            report = _summarize_strategy_rows(rows)
            print("[策略实验] 当前进度：")
            print(f"  已完成: {step_no}/{len(candidate_indices)}")
            print(f"  已耗时: {elapsed:.1f} 秒")
            print(f"  模型更新次数: {train_count}")
            if not report.empty:
                print(report.to_string(index=False))

    result = pd.DataFrame(rows)
    report = _summarize_strategy_rows(rows)
    print("[策略实验] 最终报告：")
    if report.empty:
        print("  无策略结果。")
    else:
        print(report.to_string(index=False))

    return result
