from sklearn.metrics import accuracy_score
from .strategy import StrategyClass, StrategyResults
from sklearn.tree import DecisionTreeClassifier as DT


class DT_Strategy(StrategyClass):

    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        return super().setup(y_train, y_validation, X_train_scaled, X_validation_scaled)

    def run(self):
        for criterion in ("gini", "entropy"):
            for max_depth in range (1, 6):
                for min_samples_leaf in range (2, 6):
                    for min_samples_split in range (2, 6):
                        model = DT(criterion=criterion ,max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_leaf)
                        model.fit(self.X_train_scaled, self.y_train)
                        y_pred = model.predict(self.X_validation_scaled)
                        results = StrategyResults(accuracy=accuracy_score(self.y_validation, y_pred))
                        self.results_array.append(results)
                        if self.results is None or self.results["accuracy"] < results["accuracy"]:
                            self.results = results
                            self.best_model = model

    def show_results(self):
        if (self.results):
            print(f'Accuracy DT: {self.results["accuracy"] * 100:.2f}%')
            print(f'Best DT model: {self.best_model}')
