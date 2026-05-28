# sitecustomize.py
"""Runtime patch for realtime mode only.

This module is imported automatically by Python when the project root is on
sys.path.  It intentionally does nothing for strict backtest or training modes.

Goal:
    Keep the existing strict backtest implementation unchanged, while making
    realtime signal emission follow the same adaptive_rule_switch rule boundary:
    an adaptive_rule_switch signal is production-eligible only when
    adaptive_mode=active.

Why here:
    The current realtime runner still contains realtime-only legacy coverage
    logic.  Replacing that logic at import time avoids touching run_strategy.py
    and keeps strict backtest results intact.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys


_REALTIME_MODES = {"realtime_strategies", "realtime_strategy", "realtime"}
_TARGET_MODULE = "realtime_strategy_runner"


def _is_realtime_invocation() -> bool:
    args = [str(arg) for arg in sys.argv]
    for index, arg in enumerate(args):
        if arg == "--mode" and index + 1 < len(args):
            return args[index + 1] in _REALTIME_MODES
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1] in _REALTIME_MODES
    return False


class _RealtimeAlignmentLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self.wrapped_loader.exec_module(module)
        _patch_realtime_runner(module)


class _RealtimeAlignmentFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            spec.loader = _RealtimeAlignmentLoader(spec.loader)
            return spec
        return None


def _patch_realtime_runner(module) -> None:
    original_gate = getattr(module, "passes_production_quality_gate", None)
    extract_value = getattr(module, "_extract_reason_value", None)
    if original_gate is None or extract_value is None:
        return

    def strict_backtest_equivalent_quality_gate(
        strategy_name: str,
        raw_direction: str,
        confidence: float,
        prediction: dict,
        reason: str,
        quality_context: dict | None = None,
    ) -> tuple[bool, str]:
        if strategy_name == "adaptive_rule_switch":
            if extract_value(reason, "adaptive_mode") == "active":
                return True, "production_quality_passed;adaptive_mode_active"
            return False, "production_blocked;adaptive_rule_switch_not_active"
        return original_gate(
            strategy_name=strategy_name,
            raw_direction=raw_direction,
            confidence=confidence,
            prediction=prediction,
            reason=reason,
            quality_context=quality_context,
        )

    module.passes_production_quality_gate = strict_backtest_equivalent_quality_gate
    module.REALTIME_STRICT_BACKTEST_ALIGNMENT_PATCHED = True


if _is_realtime_invocation():
    if _TARGET_MODULE in sys.modules:
        _patch_realtime_runner(sys.modules[_TARGET_MODULE])
    else:
        sys.meta_path.insert(0, _RealtimeAlignmentFinder())
