import pandas as pd
from matplotlib import pyplot as plt

pd.set_option("display.precision", 2)
from EDA import EDA
import seaborn as sb
sb.set(color_codes=True)
import numpy as np
np.random.seed(5525)
from sklearn.datasets import make_regression,make_classification,make_blobs
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lecture 4 - Feb/06/2023 - CS-5525')
    url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Train_UWu5bXk.csv'
    df = pd.read_csv(url)




    print(f'The total number of'
          f'missing entries are {df.isna().sum().sum()}')
    print('Raw')
    print(df.isna().sum()/len(df)*100)

    # Clean it
    threshold = 15
    a = df.isna().sum()/len(df)*100
    feat_names = list(df.columns)
    buffer = [None] * 10
    variable = []
    for i,fn in enumerate(feat_names):
        feat = df[fn]
        if a[i]<=threshold :
            variable.append(fn)
    df_clean = df[variable]

    print('Cleaned')
    print(df_clean.isna().sum() / len(df) * 100)


    # Fisrt normalize features
    # numeric = df_clean.select_dtypes(include=np.number)
    # type = ['int16','float16','int32','int64','float32','float64']
    # num_by_type = df.select_dtypes(include=type)
    #
    # df_normalized = eda.normalize(numeric)
    # df_filtered = eda.get_low_variance_filter(df_normalized)
    #
    # threshold = 1e-3
    # variable = []
    # feat_names = list(df_normalized.columns)
    # for i, fn in enumerate(feat_names):
    #     feat = df[fn]
    #     if df_filtered[i] <= threshold:
    #         variable.append(i)
    #
    # normalize_clean = numeric.drop(numeric.columns[variable],axis=1)
    #
    #
    # # Then calculate variance
    # df_filtered = eda.get_low_variance_filter(df_normalized)
    N =1000 # of observations
    variance2 = np.sqrt(2)
    x = np.random.normal(0,variance2,N)
    variance3 = np.sqrt(3)
    y = np.random.normal(0, variance3 ,N)

    X = np.vstack((x,y)).T

    df = pd.DataFrame(X)
    df = df- df.mean()
    # @ is multiplaction
    cov = (df.values.T @ df.values )/(len(df)-1)
    print(cov)
    # diagonal should be 2 and 3 respectfully, off diagonal should
    # be low covariance because the 2 datasets are completely independent

    # Covariance matrix to Correlation?
    # look in my notes

    # Discretization
    def change(x):
        if x<20:
            return 'young'
        elif (x<50) and (x>20):
            return 'mature'
        elif x>50:
            return 'old'


    age = [10, 11 ,13 ,32 ,34 ,40 ,72 ,73 ,75]
    df = pd.DataFrame(np.array(age), columns=['age'])
    df['description'] = df['age'].apply(change)
    print(df)

    # PCA
    n_features  =20
    X,y = make_classification(n_samples=1000,
                            n_features=n_features,
                            n_informative=20,
                            n_redundant=0,
                            n_repeated =0,
                            n_classes=4,
                            random_state=5525)# must be fixed to 5525
    # name features
    X_2 = pd.DataFrame(X,columns=[f'feature{x}' for x in range(1,X.shape[1]+1)])
    y_2 = pd.DataFrame(y,columns=['target'])
    df = pd.concat([X_2,y_2],axis=1)

    df_normalized = eda.normalize(df)
    df_normalized = df_normalized[:20]
    # normalize the data
    # df_pca = eda.get_pca(df_normalized)
    model = RandomForestRegressor(random_state=1,
                                  max_depth=100,
                                  n_estimators=100)
    model.fit(X,y)
    features = df.columns
    importance = model.feature_importances_
    indices = np.argsort(importance)[-20:]# top 20
    # plt.yticks((range(len(indices))))


    # Lab 2
    df = pd.read_csv('stock prices.csv')
    pass









