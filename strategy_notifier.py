# strategy_notifier.py

import requests

from config import ENABLE_WECHAT_NOTIFICATIONS, WECHAT_REQUEST_TIMEOUT, WECHAT_WEBHOOK_URL


STRATEGY_DISPLAY_NAMES = {
    "short_momentum": "⚡ 策略一：short_momentum 做空动量策略",
    "relaxed_scenario": "🧭 策略二：relaxed_scenario 放宽场景策略",
    "historical_match": "🔍 historical_match 历史相似样本策略",
    "historical_match_long": "🔎 historical_match_long 历史匹配做多策略",
    "historical_match_short": "🔎 historical_match_short 历史匹配做空策略",
    "kronos_confirm": "🧠 kronos_confirm Kronos 确认策略",
    "finstar_scenario": "🌟 finstar_scenario 场景推理策略",
}


def strategy_display_name(strategy_name: str) -> str:
    return STRATEGY_DISPLAY_NAMES.get(strategy_name, f"📌 {strategy_name}")


def post_wecom_markdown(lines: list[str]) -> bool:
    if not ENABLE_WECHAT_NOTIFICATIONS:
        return False
    payload = {"msgtype": "markdown", "markdown": {"content": "\n".join(lines)}}
    try:
        resp = requests.post(WECHAT_WEBHOOK_URL, json=payload, timeout=WECHAT_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            print(f"[strategy alert] wecom status={resp.status_code}, body={resp.text}")
            return False
        return True
    except Exception as exc:
        print(f"[strategy alert] send failed: {type(exc).__name__}: {exc}")
        return False


def _direction_text(direction: str) -> str:
    if direction == "up":
        return "📈 看涨 UP"
    if direction == "down":
        return "📉 看跌 DOWN"
    return "⏸️ 不预测 NO_TRADE"


def _result_text(is_correct: bool) -> str:
    return "✅ 正确" if is_correct else "❌ 错误"


def _edge_text(edge: float) -> str:
    if edge > 0:
        return f"📈 方向优势 edge(up-down)：{edge:.4f}"
    if edge < 0:
        return f"📉 方向优势 edge(up-down)：{edge:.4f}"
    return f"➖ 方向优势 edge(up-down)：{edge:.4f}"


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "暂无"
    return f"{value:.2%}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "0"
    return str(value)


def send_prediction_signal(
    strategy_name: str,
    direction: str,
    confidence: float,
    current_price: float,
    timestamp: str,
    reason: str,
    prediction_id: str,
    up_signal_probability: float | None = None,
    down_signal_probability: float | None = None,
    direction_edge: float | None = None,
    horizon_minutes: int | None = None,
    strategy_accuracy: float | None = None,
    strategy_correct_count: int | None = None,
    strategy_total_count: int | None = None,
) -> bool:
    if direction not in {"up", "down"}:
        return False

    display_name = strategy_display_name(strategy_name)
    title_emoji = "🚨📈" if direction == "up" else "🚨📉"
    lines = [
        f"{title_emoji}【BTC/USDT 高置信实时预测】",
        f"🧩 策略名称：{display_name}",
        f"🆔 预测ID：{prediction_id}",
        f"🎯 最终方向：{_direction_text(direction)}",
        f"🔥 策略置信度：{confidence:.4f}",
        f"📌 该策略正式通知历史胜率：{_fmt_rate(strategy_accuracy)}",
        f"🧾 该策略已验证正式通知样本：{_fmt_int(strategy_correct_count)}/{_fmt_int(strategy_total_count)}",
        f"💰 当前价格：{current_price:.2f}",
        f"⏰ 信号时间：{timestamp}",
        f"📝 触发原因：{reason}",
    ]

    if horizon_minutes is not None:
        lines.append(f"⏳ 验证周期：{horizon_minutes} 分钟后")

    lines.append("")
    lines.append("🤖【双子模型各自预测】")
    if up_signal_probability is not None:
        lines.append(f"📈 上涨子模型 up_model：{up_signal_probability:.4f}")
    if down_signal_probability is not None:
        lines.append(f"📉 下跌子模型 down_model：{down_signal_probability:.4f}")
    if direction_edge is not None:
        lines.append(_edge_text(direction_edge))

    lines.append("")
    lines.append("🎯 目标：高置信方向准确率。")
    return post_wecom_markdown(lines)


def send_validation_signal(
    strategy_name: str,
    prediction_id: str,
    predicted_direction: str,
    actual_direction: str,
    is_correct: bool,
    signal_price: float,
    validation_price: float,
    signal_time: str,
    validation_time: str,
    confidence: float,
    up_signal_probability: float | None = None,
    down_signal_probability: float | None = None,
    direction_edge: float | None = None,
    strategy_accuracy: float | None = None,
    strategy_correct_count: int | None = None,
    strategy_total_count: int | None = None,
) -> bool:
    display_name = strategy_display_name(strategy_name)
    result_emoji = "✅" if is_correct else "❌"
    lines = [
        f"{result_emoji}🔎【BTC/USDT 高置信预测验证】",
        f"🧩 策略名称：{display_name}",
        f"🆔 预测ID：{prediction_id}",
        f"🎯 预测方向：{_direction_text(predicted_direction)}",
        f"📊 实际方向：{_direction_text(actual_direction)}",
        f"🧾 验证结果：{_result_text(is_correct)}",
        f"💰 信号价格：{signal_price:.2f}",
        f"💵 验证价格：{validation_price:.2f}",
        f"⏰ 信号时间：{signal_time}",
        f"🕒 验证时间：{validation_time}",
        f"🔥 原始策略置信度：{confidence:.4f}",
        f"📌 该策略正式通知准确率：{_fmt_rate(strategy_accuracy)}",
        f"🧾 该策略已验证正式通知样本：{_fmt_int(strategy_correct_count)}/{_fmt_int(strategy_total_count)}",
    ]

    lines.append("")
    lines.append("🤖【原始双子模型输出】")
    if up_signal_probability is not None:
        lines.append(f"📈 上涨子模型 up_model：{up_signal_probability:.4f}")
    if down_signal_probability is not None:
        lines.append(f"📉 下跌子模型 down_model：{down_signal_probability:.4f}")
    if direction_edge is not None:
        lines.append(_edge_text(direction_edge))

    lines.append("")
    lines.append("🎯 目标：高置信方向准确率。")
    return post_wecom_markdown(lines)
