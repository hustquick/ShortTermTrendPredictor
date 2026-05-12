# main.py

import argparse

from config import (
    BACKTEST_DAYS,
    BACKTEST_MAX_STEPS,
    BACKTEST_MIN_TRAIN_SAMPLES,
    BACKTEST_MODEL_UPDATE_MINUTES,
    BACKTEST_PROGRESS_EVERY,
    BACKTEST_STEP_MINUTES,
    BACKTEST_TRAIN_WINDOW_MINUTES,
    STRICT_PARAM_SEARCH_CSV,
    STRICT_PARAM_SEARCH_ENABLED,
    STRICT_PARAM_SEARCH_TOP_N,
)
from data_download import get_recent_klines_with_cache
from run_strategy import (
    high_confidence_report,
    leaked_training_backtest,
    probability_bin_report,
    strict_walk_forward_backtest,
    threshold_search_report,
)
from strict_param_search import (
    recommend_strict_parameters,
    strict_parameter_search_report,
)


def run_train():
    """
    仅训练实时模型。
    """
    from realtime import train_model_now

    train_model_now()


def run_training_backtest(no_update_cache: bool = False):
    """
    第一阶段训练回测：允许未来数据参与训练，只看局部拟合效果。
    """
    print("[训练回测] 准备最近 48 小时历史数据...")

    try:
        df = get_recent_klines_with_cache(
            minutes=48 * 60,
            update_if_needed=not no_update_cache,
        )
    except Exception as exc:
        print(f"[训练回测] 获取历史数据失败: {type(exc).__name__}: {exc}")
        return

    result = leaked_training_backtest(df)
    report = high_confidence_report(result)

    print("[训练回测] 完成。该结果允许未来数据泄露，只代表局部拟合效果。")
    print(f"  总预测数: {report['total_rows']}")
    print(f"  有效信号数: {report['valid_signals']}")
    print(f"  有效信号胜率: {report['valid_win_rate']}")
    if not result.empty:
        print(result.tail(20).to_string(index=False))


def run_realtime():
    """
    启动实时预测。
    """
    from realtime import realtime_loop

    realtime_loop()


def run_strict_backtest(
    backtest_days: int | None = None,
    step_minutes: int | None = None,
    model_update_minutes: int | None = None,
    max_steps: int | None = None,
    no_update_cache: bool = False,
):
    """
    严格时序回测。

    数据逻辑：
    - 优先读取 data/BTCUSDT_1m_history.csv；
    - 如果本地数据不存在，则下载；
    - 如果本地数据不足或不是最新，则补充缺失区间；
    - 如果传入 --no-update-cache，则只使用本地已有数据，不主动联网补充。

    回测逻辑：
    - 固定逐根 1m K 线滚动验证；
    - 每 1 分钟产生一次预测；
    - 预测目标仍为未来第 10 分钟 close 高于或低于当前 close。
    """
    if backtest_days is None:
        backtest_days = BACKTEST_DAYS

    if step_minutes is None:
        step_minutes = BACKTEST_STEP_MINUTES

    if step_minutes != 1:
        print(
            "[严格回测] 注意：严格回测已固定为每 1 分钟逐点验证，"
            f"忽略传入步长：{step_minutes}。"
        )
        step_minutes = 1

    if model_update_minutes is None:
        model_update_minutes = BACKTEST_MODEL_UPDATE_MINUTES

    if max_steps is None:
        max_steps = BACKTEST_MAX_STEPS

    backtest_minutes = backtest_days * 24 * 60

    print("[严格回测] 准备历史数据...")
    print("[严格回测] 回测方式：每 1 分钟预测一次，逐根 1m K 线滚动验证。")
    print(f"[严格回测] 回测天数：{backtest_days} 天")
    print(f"[严格回测] 需要分钟数：{backtest_minutes}")
    print(f"[严格回测] 步长：{step_minutes} 分钟")
    print(f"[严格回测] 模型更新间隔：{model_update_minutes} 分钟")
    print(f"[严格回测] 最大回测点数：{max_steps}")
    print(f"[严格回测] 是否禁止更新缓存：{no_update_cache}")

    try:
        df = get_recent_klines_with_cache(
            minutes=backtest_minutes,
            update_if_needed=not no_update_cache,
        )
    except Exception as exc:
        print(f"[严格回测] 获取历史数据失败: {type(exc).__name__}: {exc}")
        return

    if df.empty:
        print("[严格回测] 数据为空。")
        return

    print(f"[严格回测] 数据量：{len(df)}")

    result = strict_walk_forward_backtest(
        df,
        train_window_minutes=BACKTEST_TRAIN_WINDOW_MINUTES,
        step_minutes=step_minutes,
        model_update_minutes=model_update_minutes,
        min_train_samples=BACKTEST_MIN_TRAIN_SAMPLES,
        max_steps=max_steps,
        progress_every=BACKTEST_PROGRESS_EVERY,
    )

    report = high_confidence_report(result)

    print("[严格回测] 最终报告：")
    print(f"  总预测数: {report['total_rows']}")
    print(f"  有效信号数: {report['valid_signals']}")
    print(f"  no_trade 数: {report['no_trade_rows']}")
    print(f"  有效信号占比: {report['valid_signal_ratio']}")
    print(f"  有效信号胜率: {report['valid_win_rate']}")
    print(f"  做多信号数: {report['long_signals']}")
    print(f"  做多胜率: {report['long_win_rate']}")
    print(f"  做空信号数: {report['short_signals']}")
    print(f"  做空胜率: {report['short_win_rate']}")

    if not result.empty:
        print("[严格回测] 最近 20 条结果：")
        print(result.tail(20).to_string(index=False))

        print("[严格回测] 概率分桶统计：")
        bin_report = probability_bin_report(result)
        if bin_report.empty:
            print("  无分桶结果。")
        else:
            print(bin_report.to_string(index=False))

        print("[严格回测] 阈值搜索结果：")
        threshold_report = threshold_search_report(result)
        if threshold_report.empty:
            print("  无阈值搜索结果。")
        else:
            print(threshold_report.head(30).to_string(index=False))

        if STRICT_PARAM_SEARCH_ENABLED:
            print("[严格回测] 参数组合自动搜索结果：")
            strict_param_report = strict_parameter_search_report(result)
            if strict_param_report.empty:
                print("  无参数组合搜索结果。")
            else:
                strict_param_report.to_csv(STRICT_PARAM_SEARCH_CSV, index=False)
                print(strict_param_report.head(STRICT_PARAM_SEARCH_TOP_N).to_string(index=False))
                print(f"[严格回测] 参数组合搜索结果已保存：{STRICT_PARAM_SEARCH_CSV}")

                recommendation = recommend_strict_parameters(strict_param_report)
                print("[严格回测] 参数推荐判断：")
                if recommendation["has_recommendation"]:
                    print("  结论：发现满足最低要求的候选参数。")
                    print(f"  推荐做多阈值: {recommendation['recommended_long_threshold']}")
                    print(f"  推荐做空阈值: {recommendation['recommended_short_threshold']}")
                    print(f"  总胜率: {recommendation['win_rate']}")
                    print(f"  有效信号数: {recommendation['valid_signals']}")
                    print(f"  有效信号占比: {recommendation['valid_signal_ratio']}")
                    print(f"  做多信号数: {recommendation['long_signals']}")
                    print(f"  做多胜率: {recommendation['long_win_rate']}")
                    print(f"  做空信号数: {recommendation['short_signals']}")
                    print(f"  做空胜率: {recommendation['short_win_rate']}")
                    print(f"  综合分数: {recommendation['score']}")
                    print(f"  原因: {recommendation['reason']}")
                    print("  注意：系统不会自动改写 config.py，请结合更长周期回测后手动决定。")
                else:
                    print("  结论：暂不建议更新正式交易参数。")
                    print(f"  原因: {recommendation['reason']}")
                    if "best_observed_long_threshold" in recommendation:
                        print(f"  观察到的最高排序做多阈值: {recommendation['best_observed_long_threshold']}")
                        print(f"  观察到的最高排序做空阈值: {recommendation['best_observed_short_threshold']}")
                        print(f"  观察到的胜率: {recommendation['best_observed_win_rate']}")
                        print(f"  观察到的有效信号数: {recommendation['best_observed_valid_signals']}")
                        print(f"  观察到的有效信号占比: {recommendation['best_observed_valid_signal_ratio']}")
        else:
            print("[严格回测] 参数组合自动搜索已关闭。")

    valid_signals = report["valid_signals"]

    print("[严格回测] 样本量判断：")

    if valid_signals < 10:
        print("  有效信号少于 10 个，胜率几乎没有统计意义。")
    elif valid_signals < 30:
        print("  有效信号少于 30 个，只能作为观察结果，不能作为模型有效性依据。")
    elif valid_signals < 50:
        print("  有效信号达到初步观察水平，但仍然偏少。")
    elif valid_signals < 100:
        print("  有效信号达到较有参考价值的水平。")
    else:
        print("  有效信号超过 100 个，胜率统计开始具有较强参考意义。")


def main():
    parser = argparse.ArgumentParser(
        description="ShortTermTrendPredictor BTC/USDT 1m 未来 10 分钟 close 方向预测系统"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="realtime",
        choices=["train", "realtime", "training_backtest", "strict_backtest"],
        help="运行模式：train / realtime / training_backtest / strict_backtest",
    )

    parser.add_argument(
        "--backtest-days",
        type=int,
        default=None,
        help="严格回测拉取最近多少天数据，例如 7 或 14。",
    )

    parser.add_argument(
        "--step-minutes",
        type=int,
        default=None,
        help="兼容参数。严格回测已固定为每 1 分钟逐点验证，传入其他值会被忽略。",
    )

    parser.add_argument(
        "--model-update-minutes",
        type=int,
        default=None,
        help="严格回测模型滚动更新间隔，默认等于实时重训间隔 30 分钟。",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="严格回测最大预测点数，例如 500、1000、2000。",
    )

    parser.add_argument(
        "--no-update-cache",
        action="store_true",
        help="严格回测时只使用本地历史数据，不联网补充最新数据。",
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_train()

    elif args.mode == "realtime":
        run_realtime()

    elif args.mode == "training_backtest":
        run_training_backtest(no_update_cache=args.no_update_cache)

    elif args.mode == "strict_backtest":
        run_strict_backtest(
            backtest_days=args.backtest_days,
            step_minutes=args.step_minutes,
            model_update_minutes=args.model_update_minutes,
            max_steps=args.max_steps,
            no_update_cache=args.no_update_cache,
        )

    else:
        raise ValueError(f"未知运行模式：{args.mode}")


if __name__ == "__main__":
    main()
