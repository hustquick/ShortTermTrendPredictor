import argparse
import contextlib
import itertools
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import (
    BACKTEST_MIN_TRAIN_SAMPLES,
    BACKTEST_TRAIN_WINDOW_MINUTES,
    DATA_DIR,
    OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST,
    PREDICT_HORIZON_MINUTES,
)
from core.feature_pipeline import FeaturePipeline
from core.learning_gate import RollingLearningGate
from core.risk_gate import RiskGate
from data_download import get_recent_klines_with_cache, ms_to_beijing_time
from historical_match_filter import build_historical_match_rows
from realtime_strategy_runner import (
    STRATEGY_MAP,
    _build_quality_context,
    _skipped_kronos_result,
    passes_production_quality_gate,
)
from trainer import _ensemble_predict_proba, train_validation_model
import strategies.rules as strategy_rules


STRATEGIES = (
    "short_momentum",
    "adaptive_rule_switch",
    "adaptive_dual",
    "relaxed_scenario",
    "historical_match",
    "historical_match_long",
    "historical_match_short",
    "kronos_confirm",
    "kronos_lead",
    "finstar_scenario",
)


class FastAdaptiveRuleSwitchStrategy(strategy_rules.AdaptiveRuleSwitchStrategy):
    """Indexed drop-in variant for long streaming backtests.

    The production strategy keeps the same rule selection logic but repeatedly
    scans the full in-memory record list. This variant indexes observed rows by
    rule and reuses the same filters, thresholds, and Wilson calculations.
    """

    def __init__(self):
        super().__init__()
        self._records_by_rule: dict[str, list[dict]] = defaultdict(list)
        self._indexed_upto = 0

    def _sync_index(self) -> None:
        for row in self.records[self._indexed_upto:]:
            self._records_by_rule[str(row.get("rule", ""))].append(row)
        self._indexed_upto = len(self.records)

    def _rule_records(self, rule_name: str) -> list[dict]:
        self._sync_index()
        return self._records_by_rule.get(rule_name, [])

    def _recent_rule_records(
        self,
        rule_name: str,
        timestamp_ms: float | None = None,
        lookback_days: int | None = None,
    ) -> list[dict]:
        rows = self._rule_records(rule_name)
        if timestamp_ms and timestamp_ms > 0:
            days = lookback_days or strategy_rules.ADAPTIVE_RULE_MINER_LOOKBACK_DAYS
            cutoff = int(timestamp_ms) - int(days) * 86_400_000
            rows = [row for row in rows if int(row.get("timestamp_ms") or 0) >= cutoff]
        if strategy_rules.ADAPTIVE_RULE_MINER_LOOKBACK_SIGNALS > 0:
            rows = rows[-strategy_rules.ADAPTIVE_RULE_MINER_LOOKBACK_SIGNALS:]
        return rows

    def _mine_rule_stats(
        self,
        rule_name: str,
        direction: str,
        context_tokens: tuple[str, ...],
        required_condition_tokens: tuple[str, ...] = (),
        timestamp_ms: float | None = None,
    ) -> dict | None:
        if not strategy_rules.ADAPTIVE_RULE_MINER_ENABLED or not self.records:
            return None

        min_samples = strategy_rules.ADAPTIVE_RULE_MINER_MIN_SAMPLES
        min_win_rate = strategy_rules.ADAPTIVE_RULE_MINER_MIN_WIN_RATE
        min_wilson_lower = strategy_rules.ADAPTIVE_RULE_MINER_MIN_WILSON_LOWER
        if direction == "up":
            min_samples = max(min_samples, strategy_rules.ADAPTIVE_RULE_MINER_LONG_MIN_SAMPLES)
            min_win_rate = max(min_win_rate, strategy_rules.ADAPTIVE_RULE_MINER_LONG_MIN_WIN_RATE)
            min_wilson_lower = max(min_wilson_lower, strategy_rules.ADAPTIVE_RULE_MINER_LONG_MIN_WILSON_LOWER)

        best = None
        max_clauses = max(1, strategy_rules.ADAPTIVE_RULE_MINER_MAX_CLAUSES)
        lookback_days_options = tuple(strategy_rules.ADAPTIVE_RULE_MINER_LOOKBACK_DAY_OPTIONS) or (
            strategy_rules.ADAPTIVE_RULE_MINER_LOOKBACK_DAYS,
        )
        for lookback_days in lookback_days_options:
            window = [
                row for row in self._recent_rule_records(rule_name, timestamp_ms, lookback_days=lookback_days)
                if row.get("state_ok")
                and self._tokens_match_condition(row.get("tokens", ()), required_condition_tokens)
            ]
            if len(window) < min_samples:
                continue
            search_tokens = self._miner_search_tokens(context_tokens, window, min_samples)
            optional_tokens = tuple(token for token in search_tokens if token not in required_condition_tokens)
            if not optional_tokens:
                continue

            combo_stats: dict[tuple[str, ...], list[int]] = {}
            max_combo_size = min(max_clauses, len(optional_tokens))
            for row in window:
                row_tokens = set(row.get("tokens", ()))
                correct = bool(row.get("correct"))
                if required_condition_tokens:
                    stat = combo_stats.setdefault(tuple(sorted(required_condition_tokens)), [0, 0, 0, len(required_condition_tokens)])
                    stat[0] += 1
                    stat[1] += int(correct)
                    stat[2] = 0 if correct else stat[2] + 1
                present = tuple(token for token in optional_tokens if token in row_tokens)
                if not present:
                    continue
                for clause_count in range(1, min(max_combo_size, len(present)) + 1):
                    for combo in itertools.combinations(present, clause_count):
                        condition_tokens = tuple(sorted(set(required_condition_tokens).union(combo)))
                        stat = combo_stats.setdefault(condition_tokens, [0, 0, 0, len(condition_tokens)])
                        stat[0] += 1
                        stat[1] += int(correct)
                        stat[2] = 0 if correct else stat[2] + 1

            for condition_tokens, (samples, wins, recent_loss_streak, clause_count) in combo_stats.items():
                if samples < min_samples:
                    continue
                condition = self._condition_key(condition_tokens)
                if self._condition_in_cooldown(condition):
                    continue
                win_rate = wins / samples
                wilson_lower = strategy_rules._wilson_lower_bound(wins, samples)
                if win_rate < min_win_rate or wilson_lower < min_wilson_lower:
                    continue
                if recent_loss_streak > strategy_rules.ADAPTIVE_RULE_SWITCH_MAX_RECENT_LOSS_STREAK:
                    continue
                candidate = {
                    "samples": samples,
                    "wins": wins,
                    "win_rate": float(win_rate),
                    "min_win_rate": min_win_rate,
                    "wilson_lower": float(wilson_lower),
                    "min_wilson_lower": min_wilson_lower,
                    "recent_loss_streak": recent_loss_streak,
                    "context_veto": False,
                    "condition": condition,
                    "scope": "online_miner",
                    "lookback_days": int(lookback_days),
                }
                sort_key = (
                    candidate["wilson_lower"],
                    candidate["win_rate"],
                    candidate["samples"],
                    -clause_count,
                )
                if best is None or sort_key > best[0]:
                    best = (sort_key, candidate)

        if best is None:
            return None
        return best[1]

    def _rule_stats(
        self,
        rule_name: str,
        regime: str | None = None,
        session: str | None = None,
        adaptive_context: str | None = None,
        condition_tokens: tuple[str, ...] = (),
    ) -> dict:
        condition = self._condition_key(condition_tokens)
        if not self.records:
            return super()._rule_stats(rule_name, regime, session, adaptive_context, condition_tokens)

        rows = [
            row for row in self._rule_records(rule_name)
            if row.get("state_ok")
            and (regime is None or row.get("regime") == regime)
            and (session is None or row.get("session") == session)
            and (adaptive_context is None or row.get("context") == adaptive_context)
            and self._tokens_match_condition(row.get("tokens", ()), condition_tokens)
        ][-strategy_rules.ADAPTIVE_RULE_SWITCH_ROLLING_WINDOW:]
        samples = len(rows)
        if samples == 0:
            return {**self._empty_stats(), "condition": condition}
        wins = sum(bool(row.get("correct")) for row in rows)
        return {
            "samples": samples,
            "wins": wins,
            "win_rate": float(wins / samples),
            "min_win_rate": strategy_rules.ADAPTIVE_RULE_SWITCH_MIN_WIN_RATE,
            "wilson_lower": strategy_rules._wilson_lower_bound(wins, samples),
            "min_wilson_lower": strategy_rules.ADAPTIVE_RULE_SWITCH_MIN_WILSON_LOWER,
            "recent_loss_streak": self._condition_loss_streak(rows),
            "context_veto": False,
            "condition": condition,
        }


def _actual_direction(current_price: float, future_price: float) -> str:
    return "up" if future_price > current_price else "down"


def _empty_stats() -> dict:
    return {
        "rows": 0,
        "raw_signals": 0,
        "raw_wins": 0,
        "final_signals": 0,
        "final_wins": 0,
        "official_signals": 0,
        "official_wins": 0,
    }


def _rate(wins: int, count: int) -> float | None:
    return None if count == 0 else wins / count


def _reason_value(reason: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(reason).split(";"):
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def _write_summary(stats: dict, output: Path, strategy_names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for name in strategy_names:
        item = stats[name]
        rows.append({
            "strategy": name,
            **item,
            "raw_win_rate": _rate(item["raw_wins"], item["raw_signals"]),
            "final_win_rate": _rate(item["final_wins"], item["final_signals"]),
            "official_win_rate": _rate(item["official_wins"], item["official_signals"]),
        })
    summary = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return summary


def _fast_actual_direction(current_price: pd.Series, future_price: pd.Series) -> pd.Series:
    return pd.Series(
        ["up" if f > c else "down" for c, f in zip(current_price, future_price)],
        index=current_price.index,
    )


def _install_fast_historical_match() -> None:
    from historical_match_filter import (
        MATCH_EDGE_BUCKET,
        MATCH_MIN_EDGE,
        MATCH_MIN_SAMPLES,
        MATCH_MIN_SUCCESS_RATE,
        MATCH_PROBA_BUCKET,
        MATCH_RSI_BUCKET,
        MatchResult,
        _same_sign_or_zero,
    )

    def fast_evaluate_historical_match(historical_rows, current_features, prediction, candidate_direction):
        if candidate_direction not in {"up", "down"}:
            return MatchResult(False, "no_trade", 0, None, "invalid_direction")
        if historical_rows.empty:
            return MatchResult(False, candidate_direction, 0, None, "empty_history")

        edge = float(prediction.get("direction_edge", 0.0))
        if abs(edge) < MATCH_MIN_EDGE:
            return MatchResult(False, candidate_direction, 0, None, "edge_too_small")

        df = historical_rows
        p_up = float(prediction.get("up_signal_probability", 0.0))
        p_down = float(prediction.get("down_signal_probability", 0.0))
        trend = float(current_features.get("trend_agreement", 0.0))
        macd = float(current_features.get("macd_hist", 0.0))
        ret_5 = float(current_features.get("ret_5", 0.0))
        ema = float(current_features.get("ema_5_20_diff", 0.0))
        rsi = float(current_features.get("rsi_14", 50.0))
        boll = float(current_features.get("boll_position", 0.5))

        mask = (
            df["up_signal_probability"].between(p_up - MATCH_PROBA_BUCKET, p_up + MATCH_PROBA_BUCKET)
            & df["down_signal_probability"].between(p_down - MATCH_PROBA_BUCKET, p_down + MATCH_PROBA_BUCKET)
            & df["direction_edge"].between(edge - MATCH_EDGE_BUCKET, edge + MATCH_EDGE_BUCKET)
            & _same_sign_or_zero(df["trend_agreement"], trend)
            & _same_sign_or_zero(df["macd_hist"], macd)
            & _same_sign_or_zero(df["ret_5"], ret_5)
            & _same_sign_or_zero(df["ema_5_20_diff"], ema)
            & df["rsi_14"].between(rsi - MATCH_RSI_BUCKET, rsi + MATCH_RSI_BUCKET)
            & df["boll_position"].between(boll - 0.15, boll + 0.15)
        )
        matched_signals = int(mask.sum())
        if matched_signals < MATCH_MIN_SAMPLES:
            return MatchResult(False, candidate_direction, matched_signals, None, "not_enough_matched_signals")
        matched = df.loc[mask, ["close", "future_close"]]
        success_rate = float((_fast_actual_direction(matched["close"], matched["future_close"]) == candidate_direction).mean())
        if success_rate < MATCH_MIN_SUCCESS_RATE:
            return MatchResult(False, candidate_direction, matched_signals, success_rate, "success_rate_too_low")
        return MatchResult(True, candidate_direction, matched_signals, success_rate, "historical_match_confirmed")

    strategy_rules.evaluate_historical_match = fast_evaluate_historical_match


def _batch_predictions(model, feature_block: pd.DataFrame) -> dict[int, dict]:
    if feature_block.empty:
        return {}
    X = feature_block[model.feature_cols]
    valid_mask = ~X.isna().any(axis=1)
    if not valid_mask.any():
        return {}
    X = X.loc[valid_mask]
    p_up_signal = _ensemble_predict_proba(model.up_models, X)
    p_down_signal = _ensemble_predict_proba(model.down_models, X)
    edge = p_up_signal - p_down_signal
    score_sum = p_up_signal + p_down_signal
    p_up_relative = pd.Series(0.5, index=X.index, dtype=float)
    nonzero = score_sum > 1e-12
    p_up_relative.loc[nonzero] = p_up_signal[nonzero] / score_sum[nonzero]
    predictions = {}
    for row_idx, up_sig, down_sig, rel, direction_edge in zip(
        X.index,
        p_up_signal,
        p_down_signal,
        p_up_relative.to_numpy(),
        edge,
    ):
        predictions[int(row_idx)] = {
            "predicted_direction": "no_trade",
            "up_probability": float(rel),
            "up_signal_probability": float(up_sig),
            "down_signal_probability": float(down_sig),
            "direction_edge": float(direction_edge),
            "confidence": float(max(up_sig, down_sig)),
            "high_win_rate_signal": False,
            "is_valid_signal": False,
        }
    return predictions


def run_backtest(
    df: pd.DataFrame,
    strategy_names: tuple[str, ...],
    days: int,
    step_minutes: int,
    model_update_minutes: int,
    train_window_minutes: int,
    max_steps: int | None,
    progress_every: int,
    output: Path,
) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    close_by_timestamp = df.set_index("timestamp")["close"]
    feature_pipeline = FeaturePipeline()
    print("[all_strategy_streaming] building features...")
    feature_df = feature_pipeline.build(df)
    print(f"[all_strategy_streaming] features ready rows={len(feature_df)}")

    horizon = PREDICT_HORIZON_MINUTES
    test_minutes = days * 24 * 60
    test_end = len(df) - horizon - 1
    test_start = max(train_window_minutes, test_end - test_minutes)
    candidate_indices = list(range(test_start, test_end, step_minutes))
    if max_steps is not None:
        candidate_indices = candidate_indices[-max_steps:]

    print(
        "[all_strategy_streaming] plan: "
        f"days={days}, step_minutes={step_minutes}, model_update_minutes={model_update_minutes}, "
        f"train_window_minutes={train_window_minutes}, candidate_steps={len(candidate_indices)}"
    )

    stats = defaultdict(_empty_stats)
    model = None
    next_model_update_idx = None
    block_predictions: dict[int, dict] = {}
    risk_gate = RiskGate()
    learning_gate = RollingLearningGate()
    official_strategies = set(OFFICIAL_SIGNAL_STRATEGY_ALLOWLIST)
    strategy_map = dict(STRATEGY_MAP)
    strategy_map["adaptive_rule_switch"] = FastAdaptiveRuleSwitchStrategy
    strategies = [strategy_map[name]() for name in strategy_names]
    historical_rows = None
    started = time.time()

    for step_no, idx in enumerate(candidate_indices, start=1):
        point_time = ms_to_beijing_time(int(df.iloc[idx]["timestamp"]))
        if model is None or next_model_update_idx is None or idx >= next_model_update_idx:
            train_start = max(0, idx - train_window_minutes)
            train_df = df.iloc[train_start:idx].copy()
            if len(train_df) < BACKTEST_MIN_TRAIN_SAMPLES:
                continue
            train_started = time.time()
            print(
                "[all_strategy_streaming] model update start: "
                f"step={step_no}/{len(candidate_indices)}, point_time={point_time}, train_rows={len(train_df)}"
            )
            model = train_validation_model(train_df)
            next_model_update_idx = idx + model_update_minutes
            print(
                "[all_strategy_streaming] model update done: "
                f"step={step_no}/{len(candidate_indices)}, elapsed={time.time() - train_started:.1f}s"
            )
            train_features = feature_pipeline.build(train_df).dropna(subset=model.feature_cols).copy()
            historical_rows = build_historical_match_rows(train_features, model, train_df)
            next_update = next_model_update_idx
            block_indices = [
                candidate_idx
                for candidate_idx in candidate_indices[step_no - 1:]
                if candidate_idx < next_update
            ]
            block_started = time.time()
            block_predictions = _batch_predictions(model, feature_df.iloc[block_indices])
            print(
                "[all_strategy_streaming] prediction block ready: "
                f"rows={len(block_predictions)}, elapsed={time.time() - block_started:.1f}s"
            )
            for strategy in strategies:
                if hasattr(strategy, "update_history"):
                    strategy.update_history(historical_rows)
                if hasattr(strategy, "update_kronos_result"):
                    strategy.update_kronos_result(_skipped_kronos_result("kronos_not_used_in_streaming_backtest"))

        prediction = block_predictions.get(idx)
        if prediction is None:
            continue

        current_row = df.iloc[idx]
        current_price = float(current_row["close"])
        future_ms = int(current_row["timestamp"]) + horizon * 60_000
        if future_ms not in close_by_timestamp.index:
            continue
        future_price = float(close_by_timestamp.loc[future_ms])
        actual = _actual_direction(current_price, future_price)
        feature_row = feature_df.iloc[idx]

        decisions = [(strategy, strategy.decide(feature_row, prediction)) for strategy in strategies]
        quality_context = _build_quality_context([(strategy.name, decision) for strategy, decision in decisions])

        for strategy, decision in decisions:
            name = strategy.name
            row = stats[name]
            row["rows"] += 1

            p_up = float(prediction.get("up_signal_probability", 0.0))
            p_down = float(prediction.get("down_signal_probability", 0.0))
            raw_direction = (
                decision.direction
                if decision.direction in {"up", "down"}
                else ("up" if p_up >= p_down else "down")
            )
            final_direction = decision.direction
            correct = raw_direction == actual
            row["raw_signals"] += 1
            row["raw_wins"] += int(correct)

            learning = learning_gate.decide(name, raw_direction)
            if name == "adaptive_rule_switch" and _reason_value(decision.reason, "adaptive_mode") == "active":
                learning.notify = True
                learning.state = "delegated_to_rule_switch"
                learning.reason = f"learning_delegated_to_adaptive_rule_switch;{learning.reason}"
            reason = f"{decision.reason};{learning.reason}"
            quality_ok, _ = passes_production_quality_gate(
                strategy_name=name,
                raw_direction=raw_direction,
                confidence=float(decision.confidence),
                prediction=prediction,
                reason=reason,
                quality_context=quality_context,
            )
            notify_enabled = risk_gate.is_official(
                final_direction=final_direction,
                strategy_is_allowed=name in official_strategies,
                learning_notify=learning.notify,
                quality_ok=quality_ok,
            )

            if final_direction in {"up", "down"}:
                row["final_signals"] += 1
                row["final_wins"] += int(correct)
                learning_gate.observe(name, raw_direction, correct)
                if hasattr(strategy, "observe_result"):
                    strategy.observe_result(decision, correct)
            if notify_enabled:
                row["official_signals"] += 1
                row["official_wins"] += int(correct)

        if step_no % progress_every == 0 or step_no == len(candidate_indices):
            elapsed = time.time() - started
            print(f"[all_strategy_streaming] progress {step_no}/{len(candidate_indices)} elapsed={elapsed:.1f}s")
            _write_summary(stats, output, strategy_names)

    return _write_summary(stats, output, strategy_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming strict backtest summary for all strategies.")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--step-minutes", type=int, default=1)
    parser.add_argument("--model-update-minutes", type=int, default=10080)
    parser.add_argument("--train-window-minutes", type=int, default=BACKTEST_TRAIN_WINDOW_MINUTES)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--no-update-cache", action="store_true")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "all_strategy_streaming_backtest_summary.csv")
    args = parser.parse_args()

    strategy_names = tuple(name.strip() for name in args.strategies.split(",") if name.strip())
    unknown = [name for name in strategy_names if name not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}; available={STRATEGIES}")

    required_minutes = args.train_window_minutes + args.days * 24 * 60 + PREDICT_HORIZON_MINUTES + 5
    df = get_recent_klines_with_cache(
        minutes=required_minutes,
        update_if_needed=not args.no_update_cache,
    )
    _install_fast_historical_match()
    summary = run_backtest(
        df=df,
        strategy_names=strategy_names,
        days=args.days,
        step_minutes=args.step_minutes,
        model_update_minutes=args.model_update_minutes,
        train_window_minutes=args.train_window_minutes,
        max_steps=args.max_steps,
        progress_every=args.progress_every,
        output=args.output,
    )
    print(summary.to_string(index=False))
    print(f"[all_strategy_streaming] saved: {args.output}")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
