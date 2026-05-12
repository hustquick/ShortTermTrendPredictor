# strict_param_search.py

import pandas as pd

from config import (
    MIN_SIGNALS_FOR_THRESHOLD_SEARCH,
    STRICT_PARAM_RECOMMEND_MIN_SIDE_SIGNALS,
    STRICT_PARAM_RECOMMEND_MIN_SIGNAL_RATIO,
    STRICT_PARAM_RECOMMEND_MIN_SIGNALS,
    STRICT_PARAM_RECOMMEND_MIN_WIN_RATE,
    STRICT_PARAM_SEARCH_LONG_THRESHOLDS,
    STRICT_PARAM_SEARCH_SHORT_THRESHOLDS,
)


def _win_rate(df: pd.DataFrame, direction: str) -> float | None:
    """
    计算指定方向信号的胜率。
    """
    if df.empty:
        return None

    if direction == "up":
        return float((df["actual_direction"] == "up").mean())

    if direction == "down":
        return float((df["actual_direction"] == "down").mean())

    raise ValueError(f"未知方向：{direction}")


def _score_row(row: pd.Series) -> float:
    """
    给参数组合计算综合分数。

    排序逻辑：
    - 胜率是第一优先级；
    - 信号占比不能过低；
    - 做多/做空信号过度失衡时适当扣分。
    """
    win_rate = row.get("win_rate")
    signal_ratio = row.get("valid_signal_ratio")
    long_signals = row.get("long_signals", 0)
    short_signals = row.get("short_signals", 0)
    valid_signals = row.get("valid_signals", 0)

    if pd.isna(win_rate) or valid_signals <= 0:
        return -1.0

    side_balance = min(long_signals, short_signals) / max(long_signals, short_signals, 1)

    return float(
        win_rate
        + 0.03 * min(signal_ratio, 0.5)
        + 0.02 * side_balance
    )


def strict_parameter_search_report(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    基于严格时序回测已经得到的 p_up 序列，自动搜索做多/做空阈值组合。

    注意：
    - 该函数不重新训练模型；
    - 只使用严格回测过程中已经产生的预测概率和事后验证方向；
    - 因此不会把未来数据用于模型训练，只用于回测结果统计；
    - 搜索目标是找出更合适的 long_threshold / short_threshold 组合。
    """
    required_columns = {"up_probability", "actual_direction"}

    if result_df.empty or not required_columns.issubset(result_df.columns):
        return pd.DataFrame()

    rows = []
    total_rows = len(result_df)

    for long_th in STRICT_PARAM_SEARCH_LONG_THRESHOLDS:
        for short_th in STRICT_PARAM_SEARCH_SHORT_THRESHOLDS:
            long_th = float(long_th)
            short_th = float(short_th)

            if short_th >= long_th:
                continue

            long_df = result_df[result_df["up_probability"] >= long_th].copy()
            short_df = result_df[result_df["up_probability"] <= short_th].copy()
            signal_df = pd.concat([long_df, short_df], ignore_index=True)

            valid_signals = len(signal_df)
            no_trade_rows = total_rows - valid_signals

            if valid_signals > 0:
                long_correct = long_df["actual_direction"] == "up"
                short_correct = short_df["actual_direction"] == "down"
                correct_count = int(long_correct.sum() + short_correct.sum())
                win_rate = float(correct_count / valid_signals)
            else:
                win_rate = None

            valid_signal_ratio = float(valid_signals / total_rows) if total_rows else None

            row = {
                "long_threshold": round(long_th, 4),
                "short_threshold": round(short_th, 4),
                "total_rows": total_rows,
                "valid_signals": valid_signals,
                "valid_signal_ratio": valid_signal_ratio,
                "win_rate": win_rate,
                "long_signals": len(long_df),
                "long_win_rate": _win_rate(long_df, "up"),
                "short_signals": len(short_df),
                "short_win_rate": _win_rate(short_df, "down"),
                "no_trade_rows": no_trade_rows,
                "enough_signals": valid_signals >= MIN_SIGNALS_FOR_THRESHOLD_SEARCH,
            }
            rows.append(row)

    report = pd.DataFrame(rows)

    if report.empty:
        return report

    report["recommended_candidate"] = (
        (report["valid_signals"] >= STRICT_PARAM_RECOMMEND_MIN_SIGNALS)
        & (report["valid_signal_ratio"] >= STRICT_PARAM_RECOMMEND_MIN_SIGNAL_RATIO)
        & (report["win_rate"] >= STRICT_PARAM_RECOMMEND_MIN_WIN_RATE)
        & (report["long_signals"] >= STRICT_PARAM_RECOMMEND_MIN_SIDE_SIGNALS)
        & (report["short_signals"] >= STRICT_PARAM_RECOMMEND_MIN_SIDE_SIGNALS)
    )
    report["score"] = report.apply(_score_row, axis=1)

    candidates = report[report["recommended_candidate"] == True].copy()

    if not candidates.empty:
        return candidates.sort_values(
            ["score", "win_rate", "valid_signals"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    enough = report[report["enough_signals"] == True].copy()

    if not enough.empty:
        return enough.sort_values(
            ["win_rate", "valid_signals", "valid_signal_ratio"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    return report.sort_values(
        ["valid_signals", "win_rate"],
        ascending=[False, False],
    ).reset_index(drop=True)


def recommend_strict_parameters(search_report: pd.DataFrame) -> dict:
    """
    从参数搜索报告中生成保守推荐。

    该函数只给出建议，不修改 config.py。
    """
    if search_report.empty:
        return {
            "has_recommendation": False,
            "reason": "参数搜索报告为空。",
        }

    candidates = search_report[search_report["recommended_candidate"] == True].copy()

    if candidates.empty:
        best = search_report.iloc[0]
        return {
            "has_recommendation": False,
            "reason": "没有参数组合同时满足最低样本数、最低胜率、最低信号占比和双边信号数要求。",
            "best_observed_long_threshold": float(best["long_threshold"]),
            "best_observed_short_threshold": float(best["short_threshold"]),
            "best_observed_win_rate": float(best["win_rate"]),
            "best_observed_valid_signals": int(best["valid_signals"]),
            "best_observed_valid_signal_ratio": float(best["valid_signal_ratio"]),
        }

    best = candidates.iloc[0]

    return {
        "has_recommendation": True,
        "recommended_long_threshold": float(best["long_threshold"]),
        "recommended_short_threshold": float(best["short_threshold"]),
        "win_rate": float(best["win_rate"]),
        "valid_signals": int(best["valid_signals"]),
        "valid_signal_ratio": float(best["valid_signal_ratio"]),
        "long_signals": int(best["long_signals"]),
        "long_win_rate": float(best["long_win_rate"]),
        "short_signals": int(best["short_signals"]),
        "short_win_rate": float(best["short_win_rate"]),
        "score": float(best["score"]),
        "reason": "该组合满足最低样本数、最低胜率、最低信号占比和双边信号数要求。",
    }
