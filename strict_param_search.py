# strict_param_search.py

import pandas as pd

from config import (
    MIN_SIGNALS_FOR_THRESHOLD_SEARCH,
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

            rows.append(
                {
                    "long_threshold": round(long_th, 4),
                    "short_threshold": round(short_th, 4),
                    "total_rows": total_rows,
                    "valid_signals": valid_signals,
                    "valid_signal_ratio": float(valid_signals / total_rows) if total_rows else None,
                    "win_rate": win_rate,
                    "long_signals": len(long_df),
                    "long_win_rate": _win_rate(long_df, "up"),
                    "short_signals": len(short_df),
                    "short_win_rate": _win_rate(short_df, "down"),
                    "no_trade_rows": no_trade_rows,
                    "enough_signals": valid_signals >= MIN_SIGNALS_FOR_THRESHOLD_SEARCH,
                }
            )

    report = pd.DataFrame(rows)

    if report.empty:
        return report

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
