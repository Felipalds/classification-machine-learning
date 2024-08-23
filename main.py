import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


from analysis.analysis import Analysis
from models import MethodFactory, MethodEnum

def main():

    data = pd.read_csv("./data.csv")

    print(len(data.index))

    bankrupts = data.loc[data['Bankrupt?'] == 1]

    # Oversampling
    # for i in range(15):
    #     for index, row in bankrupts.iterrows():
    #         data.loc[len(data.index)] = row.tolist()

    print(len(data.index))

    # Separation of data to trainig and testing
    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']

    plt.show()

    analysis = Analysis(X, y)
    analysis.intro()
    analysis.analyze()


if __name__ == "__main__":
    main()
