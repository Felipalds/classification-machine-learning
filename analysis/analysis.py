
from typing import Any, Dict
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.factory import MethodEnum, MethodFactory
from models.strategies.strategy import StrategyResults


class Analysis:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def analyze(self, iterations_amount: int = 20):
        df_initializer: Dict = {}
        models_results: Dict[MethodEnum, list[StrategyResults]] = {method: [] for method in MethodEnum}
        best_models: Dict[MethodEnum, Any] = {method: None for method in MethodEnum}
        for i in range(iterations_amount): # Do Training Routine each 20 times
            x_train, x_temp, y_train, y_temp = train_test_split(
                self.X, self.y, test_size=0.5)

            x_validation, x_test, y_validation, y_test = train_test_split(
                x_temp, y_temp, test_size=0.5
            )
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            current_results: Dict[MethodEnum, Any] = {method: None for method in MethodEnum}
            for method in MethodEnum:
                factory = MethodFactory(method)
                strategy = factory.create_method()
                strategy.setup(y_train, y_validation, x_train_scaled, x_test_scaled)
                strategy.fit(x_test, y_test)
                if strategy.test_results is None:
                    print("Erro, sem teste")
                    exit()
                models_results[method].append(strategy.test_results)
                if (current_results[method] is None
                    or current_results[method]["accuracy"] < strategy.test_results["accuracy"]):
                    best_models[method] = strategy.best_model
                    current_results[method] = strategy.test_results
            for key in list(models_results.keys()):
                print(len(models_results[key]))
        df_initializer.update(models_results)
        df = DataFrame(df_initializer)
        indexes = [f"Results {i}" for i in range(iterations_amount)]
        df.set_index(indexes)
        details_initializer = {"Mean Accuracy" : [], "Standard Deviation": []}
        for key in list(models_results.keys()):
            details_initializer["Mean Accuracy"].append(df[key].mean())
            details_initializer["Standard Deviation"].append(df[key].std())
        details = DataFrame(details_initializer)
        df.join(details)
        df.to_csv("summary.csv")

    def intro(self):
        print("Welcome to the analysis module!")

    def get_training_results(self) -> DataFrame:
        return DataFrame({"d": ["d"]})
