from .strategies import *

class MethodFactory:
    def __init__(self, method):
        self.method = method

    def create_method(self) -> StrategyClass:
        if self.method == "KNN":
            return KNNStrategy()
        elif self.method == "MLP":
            return MLPStrategy()
        else:
            raise ValueError("Method not found")
