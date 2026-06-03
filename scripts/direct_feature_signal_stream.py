import argparse
from pathlib import Path

import pandas as pd

from config import DATA_DIR, PREDICT_HORIZON_MINUTES
from core.feature_pipeline import FeaturePipeline
from data_download import get_recent_klines_with_cache, ms_to_beijing_time


FEATURE_EXPORT_COLUMNS = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_15",
    "ret_30",
    "ret_60",
    "ema_5_20_diff",
    "ema_10_30_diff",
    "ema_20_60_diff",
    "ema_60_240_diff",
    "macd_hist",
    "macd_hist_diff",
    "rsi_6",
    "rsi_14",
    "boll_position",
    "boll_width",
    "volatility_5",
    "volatility_10",
    "volatility_30",
    "atr_14",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "close_position",
    "volume_ratio_5",
    "volume_ratio_10",
    "volume_ratio_20",
    "trade_count_ratio_10",
    "quote_volume_ratio_10",
    "taker_buy_ratio",
    "taker_buy_ratio_diff_5_10",
    "trend_agreement",
    "mtf_3m_ret_1",
    "mtf_3m_ret_2",
    "mtf_3m_ret_3",
    "mtf_3m_ret_5",
    "mtf_3m_ema_5_20_diff",
    "mtf_3m_ema_10_30_diff",
    "mtf_3m_macd_hist",
    "mtf_3m_macd_hist_diff",
    "mtf_3m_rsi_14",
    "mtf_3m_volatility_5",
    "mtf_3m_trend_agreement",
    "mtf_3m_taker_buy_ratio",
    "mtf_3m_volume_ratio_5",
    "mtf_5m_ret_1",
    "mtf_5m_ret_2",
    "mtf_5m_ret_3",
    "mtf_5m_ret_5",
    "mtf_5m_ema_5_20_diff",
    "mtf_5m_ema_10_30_diff",
    "mtf_5m_macd_hist",
    "mtf_5m_macd_hist_diff",
    "mtf_5m_rsi_14",
    "mtf_5m_volatility_5",
    "mtf_5m_trend_agreement",
    "mtf_5m_taker_buy_ratio",
    "mtf_5m_volume_ratio_5",
]


def build_stream(args: argparse.Namespace) -> pd.DataFrame:
    required_minutes = args.days * 24 * 60 + 260 + PREDICT_HORIZON_MINUTES + 5
    raw = get_recent_klines_with_cache(minutes=required_minutes, update_if_needed=not args.no_update_cache)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    features = FeaturePipeline().build(raw)
    close_by_timestamp = raw.set_index("timestamp")["close"]
    horizon_ms = PREDICT_HORIZON_MINUTES * 60_000
    features["future_price"] = (features["timestamp"] + horizon_ms).map(close_by_timestamp)
    features["future_return"] = features["future_price"] / features["close"] - 1
    features = features.dropna(subset=["future_return", *FEATURE_EXPORT_COLUMNS]).copy()

    test_end = features["timestamp"].max()
    test_start = test_end - args.days * 24 * 60 * 60_000
    features = features[(features["timestamp"] >= test_start) & (features["timestamp"] <= test_end)].copy()
    rows = []
    for _, row in features.iterrows():
        actual_direction = "up" if row["future_return"] > 0 else "down"
        base = {
            "timestamp": ms_to_beijing_time(int(row["timestamp"])),
            "current_price": float(row["close"]),
            "future_price": float(row["future_price"]),
            "future_return": float(row["future_return"]),
            "actual_direction": actual_direction,
            "up_probability": 0.5,
            "confidence": 0.5,
            "is_valid_signal": True,
            "rule": "direct_feature",
            "model_trained_at": "",
            "dt": ms_to_beijing_time(int(row["timestamp"])),
        }
        for column in FEATURE_EXPORT_COLUMNS:
            base[column] = row.get(column)
        for direction in ("up", "down"):
            rows.append(
                {
                    **base,
                    "predicted_direction": direction,
                    "direction": direction,
                    "correct": direction == actual_direction,
                    "is_correct": direction == actual_direction,
                }
            )
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build direct feature up/down candidate stream.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--no-update-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "direct_feature_signal_stream.csv")
    args = parser.parse_args()
    out = build_stream(args)
    print("[direct_feature_signal_stream] summary:")
    print(f"  rows={len(out)}")
    print(f"  output={args.output}")


if __name__ == "__main__":
    main()
