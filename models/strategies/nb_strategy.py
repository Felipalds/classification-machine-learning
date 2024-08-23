from sklearn.metrics import accuracy_score
from .strategy import StrategyClass, StrategyResults
from sklearn.naive_bayes import GaussianNB


class NB_Strategy(StrategyClass):

    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        return super().setup(y_train, y_validation, X_train_scaled, X_validation_scaled)

    def run(self):
        model = GaussianNB()
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_validation_scaled)
        acc_score = accuracy_score(self.y_validation, y_pred)
        results = StrategyResults(accuracy=acc_score)
        self.results_array.append(results)
        if self.results is None or acc_score > self.results["accuracy"]:
            self.results = results
            self.best_model = model

    def show_results(self):
        pass
