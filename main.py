import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import MethodFactory, MethodEnum

def main():

    data = pd.read_csv("./data.csv")

    # Separation of data to trainig and testing
    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']

    for i in range(20):
        # Splitting the data into training and testing
        x_train, x_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.5)

        x_validation, x_test, y_validation, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5
        )
        # Scaling the data
        # The purpose of this standardization is to ensure that each feature contributes equally to the model's performance.
        # Standardizing the features can lead to improved performance for many machine learning algorithms.
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        for method in MethodEnum:
            factory = MethodFactory(method)
            strategy = factory.create_method()
            strategy.setup(y_train, y_validation, x_train_scaled, x_test_scaled)
            strategy.run(x_test, y_test)
            strategy.show_results()
            results = strategy.get_results()



if __name__ == "__main__":
    main()
