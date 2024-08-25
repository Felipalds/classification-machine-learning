import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


from analysis.analysis import Analysis
from models import MethodFactory, MethodEnum

def main():

    data = pd.read_csv("./data.csv")
    data = data.drop(columns=[" Revenue per person",
    " Operating profit per person",
    " Allocation rate per person",
    " Average Collection Days",
    " Net Value Per Share (B)",
    " Net Value Per Share (A)",
    " Net Value Per Share (C)",
    " Persistent EPS in the Last Four Seasons",
    " Realized Sales Gross Profit Growth Rate",
    " Total Asset Turnover",
    " Inventory and accounts receivable/Net value",
    " Fixed Assets Turnover Frequency",
    " Net Worth Turnover Rate (times)",
    " Quick Asset Turnover Rate",
    " Current Asset Turnover Rate",
    " Working capitcal Turnover Rate",
    " Cash Turnover Rate",
    " Total Asset Return Growth Rate Ratio",
    " Cash flow rate",
    " Operating Expense Rate"])

    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']

    plt.hist(y, 2)
    plt.title("Normal Sample")
    plt.show()

    print(f"Columns: {len(data.columns)} Rows: {len(data.index)}")

    bankrupts = data.loc[data['Bankrupt?'] == 1]
    non_bankrupts = data.loc[data['Bankrupt?'] == 0]

    print(f"Original Bankrupts: {len(bankrupts.index)} Original Non-Bankrupts: {len(non_bankrupts.index)}")

    # Oversampling
    for i in range(12):
        for index, row in bankrupts.iterrows():
            data.loc[len(data.index)] = row.tolist()

    print(f"Columns: {len(data.columns)} Rows: {len(data.index)}")

    # Separation of data to trainig and testing
    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']

    bankrupts = data.loc[data['Bankrupt?'] == 1]
    non_bankrupts = data.loc[data['Bankrupt?'] == 0]

    print(f"Original Bankrupts: {len(bankrupts.index)} Original Non-Bankrupts: {len(non_bankrupts.index)}")

    plt.hist(y, 2)
    plt.title("Oversampled")
    plt.show()

    analysis = Analysis(X, y)
    analysis.intro()
    analysis.analyze()


if __name__ == "__main__":
    main()
