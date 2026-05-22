# core/analyzer.py

import csv
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


class SignalAnalyzer:
    """Reads validated official signals and reports directional win rate."""

    def __init__(self, official_signals_csv: Path):
        self.official_signals_csv = official_signals_csv

    def official_win_rate(self) -> dict:
        if not self.official_signals_csv.exists():
            return {"total": 0, "correct": 0, "win_rate": None}
        total = 0
        correct = 0
        with open(self.official_signals_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("validation_status") != "validated":
                    continue
                total += 1
                correct += str(row.get("is_correct", "")).lower() == "true"
        return {
            "total": total,
            "correct": correct,
            "win_rate": correct / total if total else None,
        }
