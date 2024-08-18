from .strategy import StrategyClass, StrategyResults
from sklearn.tree import DecisionTreeClassifier as DT


class DT_Strategy(StrategyClass):

    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        return super().setup(y_train, y_validation, X_train_scaled, X_validation_scaled)

    def run(self, x_test, y_test):
        for criterion in ("gini", "entropy"):
            for max_depth in range (1, 21):
                for min_samples_leaf in range (1, 21):
                    for min_samples_split in range (2, 21):
                        model = DT(criterion=criterion ,max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_leaf)
                        model.fit(self.X_train_scaled, self.y_train)
                        y_pred = model.predict(self.X_validation_scaled)
                        acc_score = model.score(self.X_train_scaled, y_pred)
                        if acc_score > self.acc_score:
                            self.acc_score = acc_score
                            self.best_model = model

    def show_results(self):
        if (self.results):
            print(f'Accuracy DT: {self.results["accuracy"] * 100:.2f}%')
            print(f'Best DT model: {self.best_model}')


    def get_results(self) -> StrategyResults | None:
        return super().get_results()
