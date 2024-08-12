import sklearn.neural_network as nn
from sklearn.metrics import accuracy_score

from .strategy import StrategyClass

class MLPStrategy(StrategyClass):

    def setup(self, y_train, y_test, X_train_scaled, X_test_scaled):
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled

        self.acc_score = 0
        self.best_model = None
        self.best_y_pred = None

    def run(self):
        for i in range(1, 21):
            model = nn.MLPClassifier(
                hidden_layer_sizes=i*3,
                activation="relu" if i % 2 == 0 else "tanh",
                max_iter=i*3,
                # learning_rate="0.0001"
            )
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            acc_score = accuracy_score(self.y_test, y_pred)
            if acc_score > self.acc_score:
                self.acc_score = acc_score
                self.best_model = model
                self.best_y_pred = y_pred

    def show_results(self):
        print(f'Accuracy MLP: {self.acc_score * 100:.2f}%')
        print(f'Best MLP model: {self.best_model}')
        print(f'Best MLP y_pred: {self.best_y_pred}')
