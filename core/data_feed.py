# core/data_feed.py

from dataclasses import dataclass

import pandas as pd

from data_download import get_recent_klines_with_cache


@dataclass
class RealtimeDataFeed:
    """BTC/USDT 1m data source wrapper."""

    minutes: int
    update_cache: bool = True

    def load(self) -> pd.DataFrame:
        return get_recent_klines_with_cache(
            minutes=self.minutes,
            update_if_needed=self.update_cache,
        )
