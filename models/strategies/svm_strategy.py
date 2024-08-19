from .strategy import StrategyClass, StrategyResults
from sklearn.svm import SVC


class SVM_Strategy(StrategyClass):

    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        return super().setup(y_train, y_validation, X_train_scaled, X_validation_scaled)

    def run(self, x_test, y_test):
        for kernel in ("linear", "rbf", "poly", "sigmoid"):
            for C in range(1, 21):
                model = SVC(kernel=kernel, C=C)
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_validation_scaled)
                acc_score = model.score(self.X_train_scaled, y_pred)
                if acc_score > self.acc_score:
                    self.acc_score = acc_score
                    self.best_model = model

    def show_results(self):
        pass

    def get_results(self) -> StrategyResults | None:
        return super().get_results()
