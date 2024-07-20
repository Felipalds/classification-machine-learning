import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from strategy import StrategyClass
from factory import MethodFactory

methods = ["KNN", "SVM"]

def main():

    data = pd.read_csv("./data.csv")

    # Separation of data to trainig and testing
    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']


    # Splitting the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scaling the data
    # The purpose of this standardization is to ensure that each feature contributes equally to the model's performance.
    # Standardizing the features can lead to improved performance for many machine learning algorithms.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for method in methods:
        factory = MethodFactory(method)
        strategy = factory.create_method()
        strategy.execute()


if __name__ == "__main__":
    main()
