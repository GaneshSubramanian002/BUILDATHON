"""Small sklearn-compatible probability ensemble used by the v2.2 gate."""
from __future__ import annotations

import numpy as np


class ProbabilityEnsembleGate:
    def __init__(self, fixed_gate, order_gate, fixed_columns, order_columns, order_share=0.25):
        self.fixed_gate = fixed_gate
        self.order_gate = order_gate
        self.fixed_columns = list(fixed_columns)
        self.order_columns = list(order_columns)
        self.order_share = float(order_share)
        self.classes_ = fixed_gate.classes_

    def predict_proba(self, frame):
        fixed = self.fixed_gate.predict_proba(frame[self.fixed_columns])
        order = self.order_gate.predict_proba(frame[self.order_columns])
        return (1 - self.order_share) * fixed + self.order_share * order

    def predict(self, frame):
        return self.classes_[np.argmax(self.predict_proba(frame), axis=1)]

