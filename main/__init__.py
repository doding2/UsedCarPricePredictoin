import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def main():
    # drop row that contains NA and metadata columns
    dataset = pd.read_csv('cars_processed_by_unknown.csv')
    dataset = dataset.dropna(axis=0)
    dataset = dataset.drop(labels=['Unnamed: 0', '링크'], axis=1)

    # plot correlation matrix
    corr = dataset.select_dtypes('number').corr()
    sns.heatmap(corr)
    plt.show()

    # rank correlation by 가격 feature and get top related features
    price_corr = corr['가격']
    price_corr = price_corr.dropna()
    price_corr = price_corr[(price_corr > 0.3) | (price_corr < -0.3)]
    price_corr = price_corr.sort_values()
    print(price_corr)

    # select features to train and test
    X_columns = price_corr.index.to_numpy()
    X = dataset[X_columns].drop(labels=['가격'], axis=1)
    y = dataset['가격']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # train with random forest regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # compute accuracy between prediction y and test y
    y_pred = model.predict(X_test)
    scores = model.score(X_test, y_test)
    print(scores)

    # compare y_pred and y_test by plotting graph
    index = range(0, y_pred.size)
    plt.plot(index, y_test, label='y_test', color='lightblue')
    plt.plot(index, y_pred, label='y_pred', color='orange')
    plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1.1),
               ncol=2, fancybox=True, shadow=False)
    plt.xlabel('index')
    plt.ylabel('price (백만)')
    plt.title("Accuracy Score: %.2f" % scores, position=(0.3, 1.0))
    plt.show()


if __name__ == '__main__':
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
    main()
