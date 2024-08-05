
# This is an Abstract Base Class from Python
# It is used to define a common interface for all the strategies
from abc import ABC, abstractmethod

from pandas.core.groupby.groupby import ArrayLike

class StrategyClass(ABC):

    @abstractmethod
    def setup(self, y_train, y_test, X_train_scaled, X_test_scaled):
        pass

    @abstractmethod
    def run(self):
        pass


    @abstractmethod
    def show_results(self):
        pass
