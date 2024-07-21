
from knn_strategy import KNNStrategy
from strategy import StrategyClass


class MethodFactory:
    def __init__(self, method):
        self.method = method

    def create_method(self) -> StrategyClass:
        if self.method == "KNN":
            return KNNStrategy()
        else:
            raise ValueError("Method not found")
