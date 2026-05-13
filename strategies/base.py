# strategies/base.py

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class StrategyDecision:
    """统一策略输出。

    direction 只允许：up / down / no_trade。
    confidence 用于记录该策略内部认为的信号强度，不参与收益计算。
    """

    direction: str
    confidence: float
    reason: str


class DirectionStrategy(Protocol):
    """高置信方向准确率策略接口。"""

    name: str

    def decide(self, features: pd.Series, prediction: dict) -> StrategyDecision:
        """根据当前特征和模型输出，返回方向决策。"""
        ...


def feature_value(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return float(value)
