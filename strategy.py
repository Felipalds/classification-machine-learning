
# This is an Abstract Base Class from Python
# It is used to define a common interface for all the strategies
from abc import ABC, abstractmethod

class StrategyClass(ABC):

    @abstractmethod
    def execute(self):
        print("Hello world from Strategy")
