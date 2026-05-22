# run_realtime_strategies.py

import argparse

from realtime_strategy_runner import run_realtime_strategies


DEFAULT_STRATEGIES = (
    "historical_match_short"
)

OBSERVATION_STRATEGIES = (
    "short_momentum,"
    "adaptive_rule_switch,"
    "adaptive_dual,"
    "relaxed_scenario,"
    "historical_match,"
    "historical_match_long,"
    "historical_match_short,"
    "kronos_confirm,"
    "kronos_lead,"
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
    parser.add_argument(
        "--no-update-cache",
        action="store_true",
        help="Use local cached klines only and do not fetch missing realtime data.",
    )
    parser.add_argument(
        "--live-chart",
        action="store_true",
        help="Open a live matplotlib rolling 30-minute strategy chart window.",
    )
    args = parser.parse_args()
    strategy_names = OBSERVATION_STRATEGIES if args.observe_all else args.strategies

    run_realtime_strategies(
        strategy_names=strategy_names,
        train_minutes=args.train_minutes,
        once=args.once,
        update_cache=not args.no_update_cache,
        live_chart=args.live_chart,
    )


if __name__ == "__main__":
    main()
