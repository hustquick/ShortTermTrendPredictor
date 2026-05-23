# Strategy Change Policy

This project treats notification-eligible strategy signals as production rules.

## Rule Changes

- Do not relax or broaden a strategy's notification conditions unless a long-horizon strict walk-forward backtest supports the change.
- The minimum acceptance bar is 75% effective-signal accuracy in multi-fold strict validation.
- A production candidate must also average at least 10 notification-eligible signals per day in every fold.
- A candidate change must report total signals, signals per day, per-fold accuracy, per-fold signal count, and signal interval before it can be used in realtime notification mode.
- Observation-only logging can be added freely, but it must not change `notify_enabled=True` behavior.

## Current Approved Gate

`adaptive_rule_switch` uses the no-volume state gate from `c8618dc`:

- `adaptive_mode=active`
- rule samples are sufficient
- rule win rate passes the adaptive rule threshold
- `state_ok=True`

The 90-day strict walk-forward validation artifact:

- `data/adaptive_rule_switch_90d_c8618dc_4fold_summary.csv`
- combined effective-signal accuracy: about 81.6%
- combined signal density: about 35.2 notification-eligible signals per day
- all four folds above 80%
- all four folds above 32 notification-eligible signals per day

Volume-derived features may be logged in the signal reason for diagnostics. They must not block
`notify_enabled=True` unless a new strict multi-fold artifact clears both the 75% accuracy bar and
the 10 signals/day/fold density bar.
