
# This is an Abstract Base Class from Python
# It is used to define a common interface for all the strategies
from abc import ABC, abstractmethod
from typing import Any, TypedDict

from sklearn.metrics import accuracy_score

class StrategyResults(TypedDict):
    accuracy: float

class StrategyClass(ABC):

    def __init__(self) -> None:
        self.results: StrategyResults | None = None
        self.results_array: list[StrategyResults] = []
        self.best_model: Any = None
        self.test_results: StrategyResults | None = None


    @abstractmethod
    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        self.y_train = y_train
        self.y_validation = y_validation
        self.X_train_scaled = X_train_scaled
        self.X_validation_scaled = X_validation_scaled

    def fit(self, x_test, y_test):
        print(f"Running {self.__class__}")
        self.run()
        self.y_pred = self.best_model.predict(x_test)
        self.test_results = StrategyResults(accuracy=accuracy_score(y_test, self.y_pred))

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def show_results(self):
        pass
