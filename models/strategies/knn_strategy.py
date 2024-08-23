from sklearn.metrics import accuracy_score
from .strategy import StrategyClass, StrategyResults
from sklearn.neighbors import KNeighborsClassifier


class KNNStrategy(StrategyClass):

    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        super().setup(y_train, y_validation, X_train_scaled, X_validation_scaled)


    def run(self):
        for i in range(1, 21):
            for metric in ("euclidean", "manhattan", "cosine", "minkowski"):
                model = KNeighborsClassifier(n_neighbors=i, metric=metric)
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_validation_scaled)
                acc_score = accuracy_score(self.y_validation, y_pred)
                results = StrategyResults(accuracy=acc_score)
                self.results_array.append(results)
                if self.results is None or acc_score > self.results["accuracy"]:
                    self.results = results
                    self.best_model = model

    def show_results(self):
        if (self.results):
            print(f'Accuracy KNN: {self.results["accuracy"] * 100:.2f}%')
            print(f'Best KNN model: {self.best_model}')
