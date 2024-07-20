
from knn_strategy import KNNStrategy
from svm_strategy import SVMStrategy
from strategy import StrategyClass


class MethodFactory:
    def __init__(self, method):
        self.method = method

    def create_method(self) -> StrategyClass:
        if self.method == "KNN":
            return KNNStrategy()
        elif self.method == "SVM":
            return SVMStrategy()
        else:
            raise ValueError("Method not found")
