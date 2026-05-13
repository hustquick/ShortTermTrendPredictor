# run_strategy_lab.py

import argparse

from config import BACKTEST_DAYS, BACKTEST_MAX_STEPS
from data_download import get_recent_klines_with_cache
from strategy_lab import run_multi_strategy_backtest


def main():
    parser = argparse.ArgumentParser(description="Run multi-strategy direction accuracy test.")
    parser.add_argument("--backtest-days", type=int, default=BACKTEST_DAYS)
    parser.add_argument("--max-steps", type=int, default=BACKTEST_MAX_STEPS)
    parser.add_argument("--no-update-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    minutes = args.backtest_days * 24 * 60
    print("[strategy_lab] preparing data")
    print(f"[strategy_lab] backtest_days={args.backtest_days}")
    print(f"[strategy_lab] minutes={minutes}")
    print(f"[strategy_lab] max_steps={args.max_steps}")
    print(f"[strategy_lab] no_update_cache={args.no_update_cache}")
    print("[strategy_lab] objective: high-confidence directional accuracy only")

    df = get_recent_klines_with_cache(
        minutes=minutes,
        update_if_needed=not args.no_update_cache,
    )

    if df.empty:
        print("[strategy_lab] empty data")
        return

    print(f"[strategy_lab] rows={len(df)}")
    run_multi_strategy_backtest(
        df,
        max_steps=args.max_steps,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
