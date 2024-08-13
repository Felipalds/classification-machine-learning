import sklearn.neural_network as nn
from sklearn.metrics import accuracy_score

from .strategy import StrategyClass, StrategyResults

class MLPStrategy(StrategyClass):

    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        self.y_train = y_train
        self.y_validation = y_validation
        self.x_train_scaled = X_train_scaled
        self.x_validation_scaled = X_validation_scaled

        self.results: StrategyResults | None = None
        self.best_model: None | nn.MLPClassifier = None
        self.best_y_pred = None

    def run(self, x_test, y_test):
        for i in range(1, 21):
            model = nn.MLPClassifier(
                hidden_layer_sizes=i*5,
                activation="relu" if i % 2 == 0 else "tanh",
                max_iter=i*3,
                # learning_rate="0.0001"
            )
            model.fit(self.x_train_scaled, self.y_train)
            y_pred = model.predict(self.x_validation_scaled)
            acc_score = accuracy_score(self.y_validation, y_pred)
            results = StrategyResults(accuracy=acc_score)
            if self.results is None or acc_score > self.results["accuracy"]:
                self.results = results
            self.best_model = model
            self.best_y_pred = y_pred
        if (self.best_model):
            y_pred = self.best_model.predict(x_test)
            acc_score = accuracy_score(y_test, y_pred)
            results = StrategyResults(accuracy=acc_score)
            self.results = results

    def test(self, x_test):
        if (self.best_model):
            y_pred = self.best_model.predict(x_test)

    def show_results(self):
        if (self.results):
            print(f'Accuracy MLP: {self.results["accuracy"] * 100:.2f}%')
            print(f'Best MLP model: {self.best_model}')
            print(f'Best MLP y_pred: {self.best_y_pred}')


    def get_results(self) -> StrategyResults | None:
        return self.results
