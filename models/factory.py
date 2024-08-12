from typing import Literal
from .strategies import *

METHODS_TYPE = (
    Literal["KNN"] | # K-Nearest Neighbors
    Literal["MLP"] | # Multilayer Perceptron
    Literal["SVM"] | # Support Vector Machine
    Literal["DT"] | # Decision Tree
    Literal["NB"] | # Naive Bayes
    Literal["P"] # Perceptron
)

METHODS_ENUM: list[METHODS_TYPE] = [
    "KNN",
    "MLP",
    "SVM",
    "DT",
    "NB",
    "P"
]

class MethodFactory:
    def __init__(self, method: METHODS_TYPE):
        self.method: METHODS_TYPE = method

    def create_method(self) -> StrategyClass:
        if self.method == "KNN":
            return KNNStrategy()
        elif self.method == "MLP":
            return MLPStrategy()
        elif self.method == "SVM":
            return SVM_Strategy()
        elif self.method == "DT":
            return DT_Strategy()
        elif self.method == "NB":
            return NB_Strategy()
        elif self.method == "P":
            return P_Strategy()
        else:
            raise ValueError("Method not found")
