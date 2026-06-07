from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

from config import BACKTEST_TRAIN_WINDOW_MINUTES, DATA_DIR, LEGACY_MODEL_UPDATE_MINUTES, PREDICT_HORIZON_MINUTES
from data_download import get_recent_klines_with_cache
from scripts.legacy_adaptive_rule_selected_stream import build_stream


STRICT_CANDIDATE_STREAM = DATA_DIR / "legacy_recovered_selected_stream_365d_plus_online_causal_delay10_expanded_candidates.csv"
STRICT_RULE_OUTCOMES = DATA_DIR / "legacy_recovered_selected_stream_365d_plus_online_causal_delay10_expanded_rule_outcomes.csv"


@contextlib.contextmanager
def _strict_legacy_env():
    old_model_type = os.environ.get("LEGACY_LIVE_MODEL_TYPE")
    old_feature_profile = os.environ.get("LEGACY_FEATURE_PROFILE")
    os.environ["LEGACY_LIVE_MODEL_TYPE"] = "legacy_dual"
    os.environ["LEGACY_FEATURE_PROFILE"] = "all"
    try:
        yield
    finally:
        if old_model_type is None:
            os.environ.pop("LEGACY_LIVE_MODEL_TYPE", None)
        else:
            os.environ["LEGACY_LIVE_MODEL_TYPE"] = old_model_type
        if old_feature_profile is None:
            os.environ.pop("LEGACY_FEATURE_PROFILE", None)
        else:
            os.environ["LEGACY_FEATURE_PROFILE"] = old_feature_profile


def generate_strict_livefixed_candidate_stream(
    *,
    days: int,
    output: Path = STRICT_CANDIDATE_STREAM,
    rule_outcome_output: Path = STRICT_RULE_OUTCOMES,
    update_cache: bool = True,
    step_minutes: int = 1,
    model_update_minutes: int = LEGACY_MODEL_UPDATE_MINUTES,
    train_window_minutes: int = BACKTEST_TRAIN_WINDOW_MINUTES,
    progress_every_steps: int = 1000,
):
    required_minutes = train_window_minutes + int(days) * 24 * 60 + PREDICT_HORIZON_MINUTES + 5
    df = get_recent_klines_with_cache(
        minutes=required_minutes,
        update_if_needed=update_cache,
    )
    with _strict_legacy_env():
        return build_stream(
            df=df,
            days=int(days),
            step_minutes=step_minutes,
            model_update_minutes=model_update_minutes,
            train_window_minutes=train_window_minutes,
            max_steps=None,
            output=output,
            progress_every_steps=progress_every_steps,
            causal_validation_delay_minutes=PREDICT_HORIZON_MINUTES,
            rule_outcome_output=rule_outcome_output,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate strict_livefixed causal-delay10 candidate stream.")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path, default=STRICT_CANDIDATE_STREAM)
    parser.add_argument("--rule-outcome-output", type=Path, default=STRICT_RULE_OUTCOMES)
    parser.add_argument("--no-update-cache", action="store_true")
    parser.add_argument("--progress-every-steps", type=int, default=1000)
    args = parser.parse_args()
    out = generate_strict_livefixed_candidate_stream(
        days=args.days,
        output=args.output,
        rule_outcome_output=args.rule_outcome_output,
        update_cache=not args.no_update_cache,
        progress_every_steps=args.progress_every_steps,
    )
    valid = out[out["is_valid_signal"].astype(str).str.lower().eq("true")] if not out.empty else out
    print("[strict_livefixed_stream] summary")
    print(f"  output={args.output}")
    print(f"  rule_outcome_output={args.rule_outcome_output}")
    print(f"  rows={len(out)}")
    print(f"  valid_signals={len(valid)}")
    if len(valid) and "is_correct" in valid.columns:
        wr = valid["is_correct"].astype(str).str.lower().eq("true").mean()
        print(f"  valid_win_rate={wr:.4f}")


if __name__ == "__main__":
    main()
