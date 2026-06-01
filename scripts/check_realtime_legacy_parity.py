import json
import sys
from pathlib import Path

import pandas as pd

from core.legacy_adaptive_coverage_gate import DEFAULT_COVERAGE_REPORT
from core.legacy_candidate_stream import (
    LEGACY_CANDIDATE_STREAM_CSV,
    LEGACY_ONLINE_CANDIDATE_RULE_OUTCOMES_CSV,
    LEGACY_ONLINE_CANDIDATE_STREAM_CSV,
)
from core.rolling_coverage_engine import RollingCoverageConfig, build_window_item, load_candidate_rows
from data_download import beijing_time_to_ms
from realtime_strategy_runner import LEGACY_ONLINE_BOOTSTRAP_STATE


def _to_ms(value) -> int:
    ts = pd.to_datetime(value)
    return beijing_time_to_ms(ts.strftime("%Y-%m-%d %H:%M:%S"))


def _failures() -> list[str]:
    failures = []
    if "causal_delay10" not in LEGACY_CANDIDATE_STREAM_CSV.name:
        failures.append(f"default candidate stream is not causal delay10: {LEGACY_CANDIDATE_STREAM_CSV.name}")
    if "causal_delay10" not in DEFAULT_COVERAGE_REPORT.name:
        failures.append(f"default coverage report is not causal delay10: {DEFAULT_COVERAGE_REPORT.name}")
    if not LEGACY_CANDIDATE_STREAM_CSV.exists():
        failures.append(f"missing static legacy stream: {LEGACY_CANDIDATE_STREAM_CSV}")
        return failures

    static = pd.read_csv(LEGACY_CANDIDATE_STREAM_CSV, usecols=["timestamp", "model_trained_at"])
    static["timestamp_dt"] = pd.to_datetime(static["timestamp"], errors="coerce")
    static["anchor_dt"] = pd.to_datetime(static["model_trained_at"], errors="coerce")
    static = static[static["timestamp_dt"].notna()].copy()
    static_anchors = static["anchor_dt"].dropna()
    if static.empty or static_anchors.empty:
        failures.append("static legacy stream has no valid timestamp/model_trained_at")
        return failures

    static_last = static["timestamp_dt"].max()
    base_anchor = static_anchors.max()
    base_anchor_ms = _to_ms(base_anchor)
    update_ms = 10080 * 60_000

    print("[parity] static stream:")
    print(f"  path={LEGACY_CANDIDATE_STREAM_CSV}")
    print(f"  rows={len(static)}, last={static_last}, last_anchor={base_anchor}")
    print(f"  coverage_report={DEFAULT_COVERAGE_REPORT}")

    if not LEGACY_ONLINE_CANDIDATE_STREAM_CSV.exists():
        print("[parity] online stream: missing; realtime will build it on first run")
        return failures

    online = pd.read_csv(LEGACY_ONLINE_CANDIDATE_STREAM_CSV)
    if online.empty:
        print("[parity] online stream: empty")
        return failures
    online["timestamp_dt"] = pd.to_datetime(online["timestamp"], errors="coerce")
    online["anchor_dt"] = pd.to_datetime(online.get("model_trained_at", ""), errors="coerce")
    online = online[online["timestamp_dt"].notna()].copy()
    if online.empty:
        failures.append("online stream exists but has no valid timestamp")
        return failures

    first_online = online["timestamp_dt"].min()
    last_online = online["timestamp_dt"].max()
    gap_minutes = (_to_ms(first_online) - _to_ms(static_last)) / 60_000
    if gap_minutes > 12 * 60:
        failures.append(f"online stream starts too late: gap_minutes={gap_minutes:.0f}")

    anchors = online["anchor_dt"].dropna().drop_duplicates().sort_values()
    if anchors.empty:
        failures.append("online stream has no model_trained_at anchors")
    for anchor in anchors:
        anchor_ms = _to_ms(anchor)
        if anchor_ms < base_anchor_ms or (anchor_ms - base_anchor_ms) % update_ms != 0:
            failures.append(f"online anchor is off the legacy weekly schedule: {anchor}")

    print("[parity] online stream:")
    print(f"  path={LEGACY_ONLINE_CANDIDATE_STREAM_CSV}")
    print(f"  rows={len(online)}, first={first_online}, last={last_online}, gap_minutes={gap_minutes:.0f}")
    print(f"  anchors={', '.join(str(a) for a in anchors.tail(6).tolist())}")
    print(f"  rule_outcomes={LEGACY_ONLINE_CANDIDATE_RULE_OUTCOMES_CSV} exists={LEGACY_ONLINE_CANDIDATE_RULE_OUTCOMES_CSV.exists()}")

    if LEGACY_ONLINE_BOOTSTRAP_STATE.exists():
        try:
            payload = json.loads(LEGACY_ONLINE_BOOTSTRAP_STATE.read_text(encoding="utf-8"))
            print("[parity] bootstrap_state:")
            for key in ("generated_start", "generated_target", "stream_latest", "base_anchor", "live_anchor"):
                print(f"  {key}={payload.get(key)}")
        except Exception as exc:
            failures.append(f"bootstrap state unreadable: {type(exc).__name__}: {exc}")

    rows = load_candidate_rows([LEGACY_CANDIDATE_STREAM_CSV, LEGACY_ONLINE_CANDIDATE_STREAM_CSV])
    if not rows.empty:
        now_dt = rows["timestamp_dt"].max()
        item = build_window_item(
            rows,
            now_dt,
            RollingCoverageConfig(beam_size=120),
            source="online_rolling_coverage",
        )
        print("[parity] active rolling coverage window:")
        if item is None:
            print(f"  now={now_dt}, active_window=None")
        else:
            print(f"  now={now_dt}")
            print(f"  window={item.get('window')}")
            print(f"  train_start={item.get('train_start')}")
            print(f"  train_end={item.get('train_end')}")
            print(f"  cover_start={item.get('cover_start')}")
            print(f"  cover_end={item.get('cover_end')}")
            print(f"  condition={item.get('condition')}")

    return failures


def main() -> int:
    failures = _failures()
    if failures:
        print("[parity] FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("[parity] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
