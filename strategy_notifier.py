# strategy_notifier.py

import requests

from config import ENABLE_WECHAT_NOTIFICATIONS, WECHAT_REQUEST_TIMEOUT, WECHAT_WEBHOOK_URL


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
) -> bool:
    if direction not in {"up", "down"}:
        return False
    direction_text = "UP" if direction == "up" else "DOWN"
    lines = [
        "BTC/USDT high-confidence prediction signal",
        f"strategy: {strategy_name}",
        f"prediction_id: {prediction_id}",
        f"direction: {direction_text}",
        f"confidence: {confidence:.4f}",
        f"current_price: {current_price:.2f}",
        f"signal_time: {timestamp}",
        f"reason: {reason}",
    ]
    if horizon_minutes is not None:
        lines.append(f"validation_horizon: {horizon_minutes} minutes")
    if up_signal_probability is not None:
        lines.append(f"up_signal_probability: {up_signal_probability:.4f}")
    if down_signal_probability is not None:
        lines.append(f"down_signal_probability: {down_signal_probability:.4f}")
    if direction_edge is not None:
        lines.append(f"direction_edge: {direction_edge:.4f}")
    lines.append("objective: high-confidence directional accuracy only")
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
) -> bool:
    result_text = "correct" if is_correct else "wrong"
    lines = [
        "BTC/USDT high-confidence prediction validation",
        f"strategy: {strategy_name}",
        f"prediction_id: {prediction_id}",
        f"predicted_direction: {predicted_direction}",
        f"actual_direction: {actual_direction}",
        f"result: {result_text}",
        f"signal_price: {signal_price:.2f}",
        f"validation_price: {validation_price:.2f}",
        f"signal_time: {signal_time}",
        f"validation_time: {validation_time}",
        f"confidence: {confidence:.4f}",
        "objective: high-confidence directional accuracy only",
    ]
    return post_wecom_markdown(lines)
