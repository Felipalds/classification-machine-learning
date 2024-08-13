from .strategy import StrategyClass


class P_Strategy(StrategyClass):

    def setup(self, y_train, y_validation, X_train_scaled, X_validation_scaled):
        return super().setup(y_train, y_validation, X_train_scaled, X_validation_scaled)

    def run(self, x_test, y_test):
        return super().run(x_test, y_test)

    def show_results(self):
        pass

    def get_results(self) -> StrategyResults | None:
        return super().get_results()
