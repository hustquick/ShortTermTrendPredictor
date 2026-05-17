# run_realtime_strategies.py

import argparse

from realtime_strategy_runner import run_realtime_strategies


DEFAULT_STRATEGIES = (
    "historical_match_short"
)

OBSERVATION_STRATEGIES = (
    "short_momentum,"
    "relaxed_scenario,"
    "historical_match,"
    "historical_match_long,"
    "historical_match_short,"
    "kronos_confirm,"
    "finstar_scenario"
)


def main():
    parser = argparse.ArgumentParser(description="Run realtime high-confidence strategy signals.")
    parser.add_argument(
        "--strategies",
        type=str,
        default=DEFAULT_STRATEGIES,
        help=f"Comma-separated strategy names. Default: {DEFAULT_STRATEGIES}",
    )
    parser.add_argument(
        "--observe-all",
        action="store_true",
        help=f"Run all strategies for observation. Equivalent to --strategies {OBSERVATION_STRATEGIES}",
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
    strategy_names = OBSERVATION_STRATEGIES if args.observe_all else args.strategies

    run_realtime_strategies(
        strategy_names=strategy_names,
        train_minutes=args.train_minutes,
        once=args.once,
    )


if __name__ == "__main__":
    main()
