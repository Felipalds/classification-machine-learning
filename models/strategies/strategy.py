
# This is an Abstract Base Class from Python
# It is used to define a common interface for all the strategies
from abc import ABC, abstractmethod
from typing import TypedDict

from pandas import DataFrame

class StrategyResults(TypedDict):
    accuracy: float

class StrategyClass(ABC):

    @abstractmethod
    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        pass

    @abstractmethod
    def run(self, x_test, y_test):
        pass

    @abstractmethod
    def show_results(self):
        pass

    @abstractmethod
    def get_results(self) -> StrategyResults | None:
        pass
