# core/notifier.py

from strategy_notifier import send_prediction_signal, send_validation_signal


class EnterpriseWechatNotifier:
    """Notification port for official strategy signals."""

    def send_prediction(self, **kwargs):
        send_prediction_signal(**kwargs)

    def send_validation(self, **kwargs):
        send_validation_signal(**kwargs)
