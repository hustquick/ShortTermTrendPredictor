# run_realtime_strategies.py

import argparse

from realtime_strategy_runner import run_realtime_strategies


DEFAULT_STRATEGIES = "short_momentum,relaxed_scenario,historical_match"


def main():
    parser = argparse.ArgumentParser(description="Run realtime high-confidence strategy signals.")
    parser.add_argument(
        "--strategies",
        type=str,
        default=DEFAULT_STRATEGIES,
        help=f"Comma-separated strategy names. Default: {DEFAULT_STRATEGIES}",
    )
    parser.add_argument(
        "--train-minutes",
        type=int,
        default=48 * 60,
        help="Recent minutes used for model training and historical matching.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one prediction and validation cycle, then exit.",
    )
    args = parser.parse_args()

    run_realtime_strategies(
        strategy_names=args.strategies,
        train_minutes=args.train_minutes,
        once=args.once,
    )


if __name__ == "__main__":
    main()
