from enum import Enum
import itertools
from typing import Any, Dict, Literal
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
from tqdm import tqdm

class MultiModels(Enum):
    SUM = "SUM"
    MAJORITY = "MAJORITY"
    BORDA_COUNT = "BORDA_COUNT"

class Analysis:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_models: Dict[MethodEnum, Any] | None = None
        self.best_results: Dict[MethodEnum, float | None] | None = None
        self.models_results: Dict[MethodEnum, list[float]] | None = None

    def analyze(self, iterations_amount: int = 2) -> None:
        df_initializer: Dict = {}
        self.best_results = {method: None for method in MethodEnum}
        self.best_models = {method: None for method in MethodEnum}
        self.models_results = {method: [] for method in MethodEnum}
        self.best_multi_models: Dict[MultiModels, None | Any] = {m: None for m in MultiModels}
        self.best_multi_results: Dict[MultiModels, None | float] = {m: None for m in MultiModels}
        self.multi_results: Dict[MultiModels, list[float]] = {m: [] for m in MultiModels}
        for i in tqdm(range(iterations_amount)): # Do Training Routine each 20 times
            self.current_models: Dict[MethodEnum, Any] = {}
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
                self.current_models[method] = strategy.best_model
                if (current_results[method] is None
                    or current_results[method]["accuracy"] < strategy.test_results["accuracy"]):
                    self.best_models[method] = strategy.best_model
                    self.best_results[method] = strategy.test_results["accuracy"]
                    current_results[method] = strategy.test_results

            models = [(model.name, self.best_models[model]) for model in self.best_models.keys()]


            # # # 1. Regra da Soma usando VotingClassifier com 'soft' voting
            sum_rule = VotingClassifier(estimators=models, voting='soft')
            sum_rule.fit(x_train, y_train)
            sum_rule_pred = sum_rule.predict(x_test)
            sum_rule_acc = accuracy_score(y_test, sum_rule_pred)
            print(f"Voting Classifier - ACC: {sum_rule_acc}")
            self.multi_results[MultiModels.SUM].append(sum_rule_acc)
            if self.best_multi_results[MultiModels.SUM] is None:
                self.best_multi_results[MultiModels.SUM] = sum_rule_acc
                self.best_multi_models[MultiModels.SUM] = sum_rule
            elif self.best_multi_results[MultiModels.SUM] < sum_rule_acc:
                self.best_multi_results[MultiModels.SUM] = sum_rule_acc
                self.best_multi_models[MultiModels.SUM] = sum_rule


            print("Sum Rule Classifier 2")

            # # 2. Majority Vote usando VotingClassifier com 'hard' voting
            majority_vote = VotingClassifier(estimators=models, voting='hard')
            majority_vote.fit(x_train, y_train)
            majority_vote_pred = majority_vote.predict(x_test)
            majority_vote_acc = accuracy_score(y_test, majority_vote_pred)
            print(f"Majority Vote Classifier - ACC: {majority_vote_acc}")
            self.multi_results[MultiModels.MAJORITY].append(sum_rule_acc)
            if self.best_multi_results[MultiModels.MAJORITY] is None:
                self.best_multi_results[MultiModels.MAJORITY] = sum_rule_acc
                self.best_multi_models[MultiModels.MAJORITY] = sum_rule
            elif self.best_multi_results[MultiModels.MAJORITY] < sum_rule_acc:
                self.best_multi_results[MultiModels.MAJORITY] = sum_rule_acc
                self.best_multi_models[MultiModels.MAJORITY] = sum_rule

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

        df_initializer.update(self.models_results)
        df = DataFrame(df_initializer)
        indexes = [f"Results {i}" for i in range(iterations_amount)]
        df.set_index([indexes,])
        mean_accuracies = []
        stdevs = []
        mean_acc_dict: Dict[MethodEnum, float] = {}
        for key in list(self.models_results.keys()):
            mean_accuracy = float(df[key].mean())
            mean_acc_dict[key] = mean_accuracy
            mean_accuracies.append(mean_accuracy)
            stdevs.append(df[key].std())
        df.loc["Mean Accuracy"] = mean_accuracies
        df.loc["Standard Deviation"] = stdevs
        df.to_csv(f"summary.csv")

        kruskal_results = kruskal(*[self.models_results[method] for method in MethodEnum])
        print("Kruskal: ", kruskal_results)

        if kruskal_results.pvalue < 0.05:
            all_whitney_pairs = itertools.combinations(self.best_models.keys(), 2)

            for pair in all_whitney_pairs:
                whitney = mannwhitneyu(self.models_results[pair[0]], self.models_results[pair[1]], method='exact', alternative='two-sided')
                print(f"{pair[0].name} vs {pair[1].name}: {whitney}")


        multi_df_intializer = {}
        multi_df_intializer.update(self.multi_results)
        multi_df = DataFrame(multi_df_intializer)
        multi_df.set_index(indexes)
        multi_mean_accuracies = []
        multi_stdevs = []
        multi_mean_acc_dict: Dict[MultiModels, float] = {}
        for key in list(self.multi_results.keys()):
            multi_mean_accuracy = float(multi_df[key].mean())
            multi_mean_acc_dict[key] = multi_mean_accuracy
            multi_mean_accuracies.append(multi_mean_accuracy)
            multi_stdevs.append(multi_df[key].std())
        multi_df.loc["Mean Accuracy"] = multi_mean_accuracies
        multi_df.loc["Standard Deviation"] = multi_stdevs
        multi_df.to_csv(f"multi-summary.csv")

        kruskal_results = kruskal(*[self.multi_results[method] for method in MultiModels])
        print("Multi - Kruskal: ", kruskal_results)

        if kruskal_results.pvalue < 0.05:
            all_whitney_pairs = itertools.combinations(self.best_multi_models.keys(), 2)

            for pair in all_whitney_pairs:
                whitney = mannwhitneyu(self.multi_results[pair[0]], self.multi_results[pair[1]], method='exact', alternative='two-sided')
                print(f"Multi - {pair[0].name} vs {pair[1].name}: {whitney}")


    def intro(self):
        print("Welcome to the analysis module!")

    def get_training_results(self) -> DataFrame:
        return DataFrame({"d": ["d"]})

    def stats_test(self):
        if self.best_models is None or self.models_results is None:
            print("Can't do statistics test without analyzing first.")
            exit()
