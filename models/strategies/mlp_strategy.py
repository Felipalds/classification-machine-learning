import sklearn.neural_network as nn
from sklearn.metrics import accuracy_score

from .strategy import StrategyClass, StrategyResults

class MLPStrategy(StrategyClass):

    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        super().setup(y_train, y_validation, X_train_scaled, X_validation_scaled)


    def run(self):
        for hidden_layer_sizes in range(1, 2):
            print(f"MLP: {hidden_layer_sizes}")
            for learning_rate in ('constant', 'invscaling', 'adaptive'):
                for max_iter in (100, 200):
                    for activation in ('identity', 'logistic', 'tanh', 'relu'):
                        model = nn.MLPClassifier(
                            hidden_layer_sizes=(hidden_layer_sizes*20, hidden_layer_sizes, 1),
                            learning_rate=learning_rate,
                            max_iter=max_iter,
                            activation=activation
                        )
                        model.fit(self.X_train_scaled, self.y_train)
                        y_pred = model.predict(self.X_validation_scaled)
                        acc_score = accuracy_score(self.y_validation, y_pred)
                        print(acc_score)
                        results = StrategyResults(accuracy=acc_score)
                        self.results_array.append(results)
                        if self.results is None or acc_score > self.results["accuracy"]:
                            self.results = results
                            self.best_model = model

    def test(self, x_test):
        if (self.best_model):
            y_pred = self.best_model.predict(x_test)

    def show_results(self):
        if (self.results):
            print(f'Accuracy MLP: {self.results["accuracy"] * 100:.2f}%')
            print(f'Best MLP model: {self.best_model}')
