# dual_model_tuner.py

import json
from datetime import datetime

import pandas as pd

from config import (
    DUAL_MODEL_PARAM_GRID,
    DUAL_MODEL_PARAMS_FILE,
    DUAL_MODEL_TUNE_MAX_TRIALS_PER_SIDE,
    DUAL_MODEL_TUNE_MIN_VALID_SIGNALS,
    DUAL_MODEL_TUNE_MIN_WIN_RATE,
    DUAL_MODEL_TUNE_SIGNAL_THRESHOLDS,
    DUAL_MODEL_TUNE_VALID_RATIO,
    DUAL_MODEL_TUNING_REPORT_CSV,
    LABEL_NEUTRAL_THRESHOLD,
    RANDOM_STATE,
)
from features import make_dual_train_dataset
from trainer import (
    _ensemble_predict_proba,
    _fit_ensemble,
    _validate_binary_target,
)


def _score_candidate(win_rate: float | None, signals: int, signal_ratio: float) -> float:
    """
    调参综合评分。

    优先提高胜率，同时要求有足够信号数。该评分只用于参数选择，
    不代表真实收益率。
    """
    if win_rate is None or signals <= 0:
        return -1.0

    signal_bonus = min(signal_ratio, 0.5) * 0.05
    sample_bonus = min(signals / max(DUAL_MODEL_TUNE_MIN_VALID_SIGNALS, 1), 2.0) * 0.01

    return float(win_rate + signal_bonus + sample_bonus)


def _evaluate_side(
    side: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict,
) -> dict:
    """
    训练并评估单个方向子模型。

    同一组模型参数下，不再固定使用 0.80 作为信号阈值，
    而是在 DUAL_MODEL_TUNE_SIGNAL_THRESHOLDS 中搜索更合适的概率阈值。
    """
    models = _fit_ensemble(X_train, y_train, mode="validation", params=params)
    probability = _ensemble_predict_proba(models, X_valid)
    total = len(X_valid)

    best = None

    for threshold in DUAL_MODEL_TUNE_SIGNAL_THRESHOLDS:
        signal_mask = probability >= threshold
        signals = int(signal_mask.sum())

        if signals > 0:
            win_rate = float(y_valid[signal_mask].mean())
        else:
            win_rate = None

        signal_ratio = float(signals / total) if total else 0.0
        score = _score_candidate(win_rate, signals, signal_ratio)

        row = {
            "side": side,
            "threshold": float(threshold),
            "signals": signals,
            "total": total,
            "signal_ratio": signal_ratio,
            "win_rate": win_rate,
            "score": score,
            "params": params,
        }

        if best is None or row["score"] > best["score"]:
            best = row

    return best


def _tune_one_side(
    side: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[dict, list[dict]]:
    """
    对 up 或 down 子模型进行网格搜索。
    """
    _validate_binary_target(y_train, f"{side}_train_label")
    _validate_binary_target(y_valid, f"{side}_valid_label")

    trials = DUAL_MODEL_PARAM_GRID[:DUAL_MODEL_TUNE_MAX_TRIALS_PER_SIDE]
    rows = []

    for trial_no, params in enumerate(trials, start=1):
        print(f"[双子模型调参] {side} 子模型 trial {trial_no}/{len(trials)}")
        result = _evaluate_side(
            side=side,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            params=params,
        )
        result["trial_no"] = trial_no
        rows.append(result)
        print(
            f"  threshold={result['threshold']}, "
            f"signals={result['signals']}, "
            f"signal_ratio={result['signal_ratio']}, "
            f"win_rate={result['win_rate']}, "
            f"score={result['score']}"
        )

    qualified = [
        row for row in rows
        if row["signals"] >= DUAL_MODEL_TUNE_MIN_VALID_SIGNALS
        and row["win_rate"] is not None
        and row["win_rate"] >= DUAL_MODEL_TUNE_MIN_WIN_RATE
    ]

    pool = qualified if qualified else rows
    best = max(pool, key=lambda row: row["score"])

    return best, rows


def tune_dual_model_params(df: pd.DataFrame) -> dict:
    """
    自动搜索双子模型参数并保存到 JSON。

    训练/验证划分严格按时间顺序进行：前段训练，后段验证。
    """
    X, y_up, y_down, data, feature_cols = make_dual_train_dataset(df)

    if len(X) < 1000:
        raise ValueError(f"调参样本过少：{len(X)}，建议至少 1000 条。")

    split_idx = int(len(X) * (1.0 - DUAL_MODEL_TUNE_VALID_RATIO))

    X_train = X.iloc[:split_idx].copy()
    X_valid = X.iloc[split_idx:].copy()

    y_up_train = y_up.iloc[:split_idx].copy()
    y_up_valid = y_up.iloc[split_idx:].copy()
    y_down_train = y_down.iloc[:split_idx].copy()
    y_down_valid = y_down.iloc[split_idx:].copy()

    print("[双子模型调参] 数据切分：")
    print(f"  总样本数: {len(X)}")
    print(f"  训练样本数: {len(X_train)}")
    print(f"  验证样本数: {len(X_valid)}")
    print(f"  标签阈值: ±{LABEL_NEUTRAL_THRESHOLD}")

    best_up, up_rows = _tune_one_side(
        side="up",
        X_train=X_train,
        y_train=y_up_train,
        X_valid=X_valid,
        y_valid=y_up_valid,
    )
    best_down, down_rows = _tune_one_side(
        side="down",
        X_train=X_train,
        y_train=y_down_train,
        X_valid=X_valid,
        y_valid=y_down_valid,
    )

    report_rows = []
    for row in up_rows + down_rows:
        flat = {
            "side": row["side"],
            "trial_no": row["trial_no"],
            "threshold": row["threshold"],
            "signals": row["signals"],
            "total": row["total"],
            "signal_ratio": row["signal_ratio"],
            "win_rate": row["win_rate"],
            "score": row["score"],
        }
        flat.update({f"param_{key}": value for key, value in row["params"].items()})
        report_rows.append(flat)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(DUAL_MODEL_TUNING_REPORT_CSV, index=False)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label_neutral_threshold": LABEL_NEUTRAL_THRESHOLD,
        "valid_ratio": DUAL_MODEL_TUNE_VALID_RATIO,
        "train_samples": len(X_train),
        "valid_samples": len(X_valid),
        "up": {
            **best_up["params"],
            "signal_threshold": best_up["threshold"],
        },
        "down": {
            **best_down["params"],
            "signal_threshold": best_down["threshold"],
        },
        "up_metric": {
            "signals": best_up["signals"],
            "signal_ratio": best_up["signal_ratio"],
            "win_rate": best_up["win_rate"],
            "score": best_up["score"],
            "threshold": best_up["threshold"],
        },
        "down_metric": {
            "signals": best_down["signals"],
            "signal_ratio": best_down["signal_ratio"],
            "win_rate": best_down["win_rate"],
            "score": best_down["score"],
            "threshold": best_down["threshold"],
        },
        "random_state": RANDOM_STATE,
    }

    with open(DUAL_MODEL_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[双子模型调参] 完成。")
    print(f"  参数文件: {DUAL_MODEL_PARAMS_FILE}")
    print(f"  报告文件: {DUAL_MODEL_TUNING_REPORT_CSV}")
    print("[双子模型调参] up 最优：")
    print(payload["up_metric"])
    print("[双子模型调参] down 最优：")
    print(payload["down_metric"])

    return payload
