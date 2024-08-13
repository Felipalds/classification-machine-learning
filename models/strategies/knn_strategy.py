from sklearn.metrics import accuracy_score
from .strategy import StrategyClass, StrategyResults
from sklearn.neighbors import KNeighborsClassifier


class KNNStrategy(StrategyClass):

    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        self.y_train = y_train
        self.y_test = y_validation
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_validation_scaled

        self.results: StrategyResults | None = None
        self.best_model = None
        self.best_y_pred = None


    def run(self, x_test, y_test):
        for i in range(1, 21):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            acc_score = accuracy_score(self.y_test, y_pred)
            if acc_score > self.acc_score:
                self.acc_score = acc_score
                self.best_model = model
                self.best_y_pred = y_pred

    def show_results(self):
        print(f'Accuracy KNN: {self.acc_score * 100:.2f}%')
        print(f'Best KNN model: {self.best_model}')
        print(f'Best KNN y_pred: {self.best_y_pred}')

    def get_results(self) -> StrategyResults | None:
        return super().get_results()
