import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analysis.analysis import Analysis
from models import MethodFactory, MethodEnum

def main():

    data = pd.read_csv("./data.csv")

    # Separation of data to trainig and testing
    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']

    analysis = Analysis(X, y)
    analysis.intro()
    analysis.analyze()


if __name__ == "__main__":
    main()
