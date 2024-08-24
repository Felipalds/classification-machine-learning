import itertools
from typing import Any, Dict
from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from scipy.stats import kruskal, mannwhitneyu
from models.factory import MethodEnum, MethodFactory
from models.strategies.strategy import StrategyResults
import numpy as np

class Analysis:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_models: Dict[MethodEnum, Any] | None = None
        self.best_results: Dict[MethodEnum, float | None] | None = None
        self.models_results: Dict[MethodEnum, list[float]] | None = None

    def analyze(self, iterations_amount: int = 20) -> None:
        df_initializer: Dict = {}
        self.best_results = {method: None for method in MethodEnum}
        self.best_models = {method: None for method in MethodEnum}
        self.models_results = {method: [] for method in MethodEnum}
        for i in range(iterations_amount): # Do Training Routine each 20 times
            x_train, x_temp, y_train, y_temp = train_test_split(
                self.X, self.y, test_size=0.5)

            x_validation, x_test, y_validation, y_test = train_test_split(
                x_temp, y_temp, test_size=0.5
            )
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            current_results: Dict[MethodEnum, Any] = {method: None for method in MethodEnum}
            for method in MethodEnum:
                factory = MethodFactory(method)
                strategy = factory.create_method()
                strategy.setup(y_train, y_validation, x_train_scaled, x_test_scaled)
                strategy.fit(x_test, y_test)
                if strategy.test_results is None:
                    print("Erro, sem teste")
                    exit()
                self.models_results[method].append(strategy.test_results["accuracy"])
                if (current_results[method] is None
                    or current_results[method]["accuracy"] < strategy.test_results["accuracy"]):
                    self.best_models[method] = strategy.best_model
                    self.best_results[method] = strategy.test_results["accuracy"]
                    current_results[method] = strategy.test_results
            for key in list(self.models_results.keys()):
                print(len(self.models_results[key]))

        df_initializer.update(self.models_results)
        df = DataFrame(df_initializer)
        indexes = [f"Results {i}" for i in range(iterations_amount)]
        df.set_index([indexes,])
        mean_accuracies = []
        stdevs = []
        for key in list(self.models_results.keys()):
            mean_accuracies.append(df[key].mean())
            stdevs.append(df[key].std())
        df.loc["Mean Accuracy"] = mean_accuracies
        df.loc["Standard Deviation"] = stdevs
        df.to_csv(f"summary.csv")

        models = [(model.name, self.best_models[model]) for model in self.best_models.keys()]

        x_train, x_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=0.5)

        x_validation, x_test, y_validation, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5
        )

        kruskal_results = kruskal(*[self.best_results[model] for model in self.best_results])
        print("Kruskal: ", kruskal_results) # FAZER COM A MÉDIA DAS ACURÁCIAS!!

        if kruskal_results.pvalue < 0.05:
            all_whitney_pairs = itertools.combinations(self.best_models.keys(), 2)

            for pair in all_whitney_pairs: # MESMA COISA, USAR MÉDIAS
                whitney = mannwhitneyu(self.best_results[pair[0]], self.best_results[pair[1]], method='exact', alternative='two-sided')
                print(f"{pair[0].name} vs {pair[1].name}: {whitney}")

            # # # # 1. Regra da Soma usando VotingClassifier com 'soft' voting
            # sum_rule = VotingClassifier(estimators=models, voting='soft')
            # sum_rule.fit(x_train, y_train)
            # sum_rule_pred = sum_rule.predict(x_test)

            # # # 2. Majority Vote usando VotingClassifier com 'hard' voting
            # majority_vote = VotingClassifier(estimators=models, voting='hard')
            # majority_vote.fit(x_train, y_train)
            # majority_vote_pred = majority_vote.predict(x_test)

            # # 3. Borda Count customizado
            # class BordaCountClassifier(BaseEstimator, ClassifierMixin):
            #     def __init__(self, estimators):
            #         self.estimators = estimators

            #     def fit(self, X, y):
            #         for name, estimator in self.estimators:
            #             estimator.fit(X, y)
            #         return self

            #     def predict(self, X):
            #         probas = np.asarray([est.predict_proba(X) for _, est in self.estimators])
            #         rankings = np.argsort(np.argsort(probas, axis=2), axis=2)
            #         borda_scores = np.sum(rankings, axis=0)
            #         return np.argmax(borda_scores, axis=1)

            # # Instanciar o Borda Count
            # borda_count = BordaCountClassifier(estimators=models)
            # borda_count.fit(x_train, y_train)
            # borda_count_pred = borda_count.predict(x_test)

            # # Avaliar as regras de combinação
            # print("Accuracy using Sum Rule:", accuracy_score(y_test, sum_rule_pred))
            # print("Accuracy using Majority Vote:", accuracy_score(y_test, majority_vote_pred))
            # print("Accuracy using Borda Count:", accuracy_score(y_test, borda_count_pred))

        # Multiplos classifiers
        # for c in range (0, 20) :
            # self.bestmodels
            # (sum)
            # (majority)
            # (bordas)
            #
            #

    # Testes estatísticos
    #   Múltiplos
    #   normais

    def intro(self):
        print("Welcome to the analysis module!")

    def get_training_results(self) -> DataFrame:
        return DataFrame({"d": ["d"]})

    def stats_test(self):
        if self.best_models is None or self.models_results is None:
            print("Can't do statistics test without analyzing first.")
            exit()
