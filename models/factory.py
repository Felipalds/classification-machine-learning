from typing import Literal
from .strategies import *
from enum import Enum

class MethodEnum(Enum):
    KNN = KNNStrategy()
    MLP = MLPStrategy()
    SVM = SVM_Strategy()
    DT = DT_Strategy()
    NB = NB_Strategy()
    P = P_Strategy()

class MethodFactory:
    def __init__(self, method: MethodEnum):
        self.method: MethodEnum = method

    def create_method(self) -> StrategyClass:
        return self.method.value
