import warnings

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np

sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

import numpy as np
import pandas as pd
pd.set_option("display.precision", 2)
import seaborn as sns
from scipy.stats import gmean
from scipy.stats import hmean
from sklearn import metrics
from sklearn.datasets import make_regression,make_classification,make_blobs
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


class EDA:
    def __init__(self, info = None):
        self.info = info

    def get_num_observations(self,df,name=None):
        return len(df)

    def show_feature_datatype(self,df,sb_name=None):
        feat_names = list(df.columns)
        qual_feats = []
        quant_feats = []
        for fn in feat_names:
            feat = df[fn]
            d_type = str(feat.dtypes)
            if d_type == "int64" or d_type == 'float64':
                quant_feats.append(fn)
            else:
                qual_feats.append(fn)
        print(f' Quantitative features:  {quant_feats}')
        print(f' Qualitative features:  {qual_feats}')




    def get_num_categorical_features(self,df,sb_name=None):
        # is a nominal feature present?

            types = df.dtypes.astype(str)
            obj_result = types.str.contains(pat='object').any()
            if obj_result:
                return df.dtypes.astype(str).value_counts()['object']

    def get_num_numerical_features(self,df,sb_name=None):
            # is a numerical feature present?

            types = df.dtypes.astype(str)
            float_result = types.str.contains(pat='float64').any()
            int_result = types.str.contains(pat='int64').any()
            obj_result = types.str.contains(pat='object').any()

            if float_result and int_result:
                return  np.sum((df.dtypes.astype(str).value_counts()['int64'],df.dtypes.astype(str).value_counts()['float64']))
            elif float_result and not int_result:
                return df.dtypes.astype(str).value_counts()['float64']
            elif int_result and not float_result:
                return df.dtypes.astype(str).value_counts()['int64']

    def get_numerical_features(self,df,sb_name=None):

        feat_names = list(df.columns)
        num_feat_names = []
        for fn in feat_names:
            feat = df[fn]
            d_type = str(feat.dtypes)
            if d_type == "int64" or d_type == 'float64':
                num_feat_names.append(fn)
        return df[num_feat_names]

    def show_agg_by_feature(self,df,name=None,printF=None,saveF=None):


        feat_names = list(df.columns)

        buffer = [None] * 10
        for fn in feat_names:
            # print(fn '   ')
            title = str(fn)
            feat = df[fn]
            d_type = str(feat.dtypes)
            if d_type == "int64" or d_type == 'float64':
                obs_count = feat.notnull().sum()
                mean = feat.mean()
                std = feat.std()
                minimum = feat.min()
                quantile_25 = feat.quantile(0.25)
                quantile_50 = feat.quantile(0.5)
                quantile_75 = feat.quantile(0.75)
                maximum = feat.max()
                missing_obs = feat.isnull().sum()
                buffer = np.vstack((buffer, [title , obs_count,mean,std,minimum,quantile_25,quantile_50,
                                         quantile_75,maximum,missing_obs]))
        df_processed = pd.DataFrame(buffer[1:], columns=['Feature Title ','Observation Count',
                                                                                'Mean','STD','Min.','25% Quartile',
                                                                                '50% Quartile','75% Quartile','Max.',
                                                                                'Missing Observations'])

        if printF:
            print(name + ' dataset ')
            print(df_processed)

            print('Nominal features: survived, sex, embarked, who,'
                  ' adult_male, deck, embark_town, alive, alone\n')
            print('Ordinal features: pclass, class, sibsp \n')
            print('Interval features: none\n')
            print('Ratio features: fare, age\n')
            print('Missing observations: ' + str(df.isnull().sum().sum())+'\n')

        if saveF:
            pass #TODO

    def get_percent_of_missing_observations(self,df,clean,clean_method, show):

        df_has_missing = df[df.isna().any(axis=1)]

        if show:
            print('The number of missing entries by feature are \n')
            print(f'{df.isna().sum()}')

            print(f'The total number of '
                  f'missing entries are {df.isna().sum().sum()}')

            if clean:
                if clean_method is None or 'mean':
                    clean_method = 'mean'
                    df_cleaned = df.fillna(df.mean(axis=0, skipna=True))

                    print('Head of cleaned dataset: \n')
                    print(df_cleaned.head())

                    print('The number of missing entries in the cleaned dataset by feature are \n')
                    print(f'{df_cleaned.isna().sum()}')

                    print(f'The total number of '
                          f'missing entries in the cleaned dataset are {df_cleaned.isna().sum().sum()}')
                    print('Percentage missing by feature \n')
                    print(df_cleaned.isna().sum() / len(df_cleaned) * 100)

            else:
                df_cleaned = None
        num_missing = len(df_has_missing)
        percent_removed = (num_missing / len(df))*1e2
        return num_missing, percent_removed, df_cleaned

    def get_means(self,df,metrics):
        # metrics.update({:})
        feat_names = list(df.columns)

        buffer = [None] * 10
        for fn in feat_names:
            title = str(fn)
            feat = df[fn]
            metrics.update({ str(title + ' arithmetic mean'): f' {np.mean(feat):.2f}'})
            metrics.update({str(title + ' geometric mean'): f' {gmean(feat):.2f}'})
            metrics.update({str(title + ' harmonic mean'): f' {hmean(feat):.2f}'})
        return metrics

    def get_low_variance_filter(self,df):
        pass

    def normalize(self,df):
        norm = normalize(df)
        # df_norm = pd.DataFrame(norm,columns = [f'feature{i}'for i in range(1,len(df.columns))])
        df_norm = pd.DataFrame(norm)
        return df_norm
        # or
        # scaler = MinMaxScaler()
        # norm =

    def get_pca(self,df):
        pca = PCA(n_components=len(df.columns),svd_solver='full')
        pca.fit(df)
        X_PCA = pca.transform(df)
        return X_PCA

        # TODO def compile_data_fram(self,data):

    def plot_features_unified_xaxis(self,df,x_axis_feature,y_axis_feature,observation_ID,show_plot,title):
        plt.figure(figsize=(12, 8))
        legFlag = True
        for obs in observation_ID:
            for i,yy in enumerate(y_axis_feature):
                x = df[df['symbol'] == obs][x_axis_feature]
                y = df[df['symbol'] == obs][y_axis_feature]
                if y.isnull().values.any():
                    data = df.dropna()
                    x = data[data['symbol'] == obs][x_axis_feature]
                    y = data[data['symbol'] == obs][y_axis_feature]
                if title == 'Google stock price and moving average' and i==1:
                    plt.plot(x, y, label=yy)
                elif title == 'Google stock price and moving average':
                    if legFlag:
                        plt.plot(x,y, label = obs)
                        legFlag=False
                else:
                    plt.plot(x, y, label=obs)




        n_samples = len(x)
        plt.xlabel(x_axis_feature, weight='bold', size=12)
        plt.ylabel(y_axis_feature, weight='bold', size=12)
        plt.title(title, weight='bold', size=12)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.grid(color='k', linestyle='-')
        plt.xticks(np.array([x.iloc[0],
                            x.iloc[int(1*(n_samples/6))],
                            x.iloc[int(2*(n_samples/6))],
                            x.iloc[int(3*(n_samples/6))],
                            x.iloc[int(4*(n_samples/6))],
                            x.iloc[int(5*(n_samples/6))],
                            x.iloc[int(6 * (n_samples / 6))-1]
                             ]))



    def slice_by_observation(self,df,feature,observations):
        cmmd_str = ''
        for i,obs in enumerate(observations):
            if i != len(observations)-1:
                cmmd_str = cmmd_str + '(df['"'" + feature + "'"'] == '"'"+ obs +"'" ') | '
            else:
                cmmd_str = cmmd_str + '(df['"'" + feature + "'"'] == ' "'"+ obs +"'" ')'

        return eval('df['+ cmmd_str +']')


if __name__ == '__main__':
    show_plot = True
    eda = EDA('Clifford Tatum LAB 2 - CS-5525')
    nl = '\n'
    ######### SUBMITAL ###############
    url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/stock%20prices.csv'
    df = pd.read_csv(url)
    # ######### SUBMITAL ###############
    # df = df[(df['date'] > '2015-01-01')]
    # df = df.iloc[:1000,:]

    # Problem 1
    n_missing, percent_missing,df_stock_cleaned = eda.get_percent_of_missing_observations(df,
                                                                                          clean=True,
                                                                                          clean_method = 'mean',
                                                                                          show=True)


    # Problem 2
    print(f'Number of unique companies: {len(df_stock_cleaned["symbol"].unique())}')
    print('Unique companies:\n')
    print(df_stock_cleaned['symbol'].unique())
    eda.show_feature_datatype(df_stock_cleaned)

    df_googl_apple_ = eda.slice_by_observation(df_stock_cleaned,
                                             feature='symbol',
                                             observations = ['AAPL','GOOGL'])

    eda.plot_features_unified_xaxis(df_googl_apple_,
                                    x_axis_feature='date',
                                    y_axis_feature=["close"],
                                    observation_ID=['AAPL', 'GOOGL'],
                                    show_plot=True,
                                    title = 'Apple and Google stock closing price comparison' )


    # Problem 3
    df_agg_symbol = df_stock_cleaned.groupby('symbol').sum()
    print(df_agg_symbol.head())




    # Problem 4
    df_sliced_scv = df_stock_cleaned[['symbol','close','volume']]
    df_agg_scv_mean = df_sliced_scv.groupby('symbol').mean()
    df_agg_scv_variance = df_sliced_scv.groupby('symbol').var()

    print(f'Mean Aggregate of stock dataset (first 5 rows){nl}{df_agg_scv_mean.head()}')
    print(f'Variance Aggregate of stock dataset (first 5 rows){nl}{df_agg_scv_variance.head()}')

    print(f'The company with the maximum variance in closing cost is '
          f'{df_agg_scv_variance["close"].idxmax()} , with a maximum variance of '
          f'{df_agg_scv_variance["close"].max()}')



    # Problem 5
    df_google= eda.slice_by_observation(df_stock_cleaned,
                                             feature='symbol',
                                             observations = ['GOOGL'])
    df_date_after = df_google[(df_google['date'] > '2015-01-01')]
    df_after_init =  df_date_after
    print(f'Google stock data after 2015-01-01:  {nl}{df_date_after.head()}')



    # Problem 6

    df_date_after['date'] = pd.to_datetime(df_date_after['date'])
    df_date_after_init = df_date_after
    df_date_after['close - SMA 30 day'] = df_date_after['close'].rolling(30).mean()
    df_sma_close = df_date_after.dropna()

    print(f'The number of observations that will be missed as a result of applying a 30 day rolling average is '
          f' {len(df_date_after)-len(df_sma_close)}')

    eda.plot_features_unified_xaxis(df_date_after,
                                    x_axis_feature='date',
                                    y_axis_feature=['close','close - SMA 30 day'],
                                    observation_ID=['GOOGL'],
                                    show_plot=True,
                                    title='Google stock price and moving average')


    # Problem 7
    data_to_cut = df_date_after['close']
    df_date_after['price category'] = pd.cut(x=data_to_cut,
                               bins=5,
                               labels=['very low', 'low', 'normal', 'high', 'very high'],
                               include_lowest=True)


    df_cat_price_dat = df_date_after[['date', 'close', 'price category']]
    print(df_cat_price_dat.to_string())

    # Problem 8

    labels =['very low', 'low', 'normal', 'high', 'very high']
    colors = ['blue', 'darkorange', 'green', 'red', 'purple']
    fig, ax = plt.subplots(figsize=(12, 8))
    df_date_after['price category'].value_counts().plot(kind='bar',
                                                        color=['blue','orange','green','red', 'purple'])
    plt.xticks(rotation=0, ha='right')
    plt.xlabel('price_category', weight='bold', size=12)
    plt.ylabel('count', weight='bold', size=12)
    plt.title('equal width discretization', weight='bold', size=12)
    plt.grid(color='k', linestyle='-')


    # Problem 9
    data_to_cut = df_date_after_init['close']
    df_cat_price_dat =df_date_after_init
    df_cat_price_dat['price category'] = pd.qcut(x = data_to_cut,q=5,
                                              labels=['very low', 'low', 'normal', 'high', 'very high'])
    df_cat_price_temp = df_cat_price_dat[['date', 'close', 'price category']]
    print(df_cat_price_temp.to_string())

    colors = ['blue', 'darkorange', 'green', 'red', 'purple']
    fig, ax = plt.subplots(figsize=(12, 8))
    df_cat_price_dat['price category'].value_counts().plot(kind='bar',
                                                           color=['blue','orange','green','red', 'purple'])
    plt.xticks(rotation=0, ha='right')
    plt.xlabel('price_category', weight='bold', size=12)
    plt.ylabel('count', weight='bold', size=12)
    plt.title('equal frequency discretization', weight='bold', size=12)
    plt.grid(color='k', linestyle='-')



    # Problem 10
    df_date_after = df_google[(df_google['date'] > '2015-01-01')]
    X = eda.get_numerical_features(df_date_after)

    N, M = X.shape
    N = eda.get_num_observations(df_date_after)
    cov_df_date_after = np.zeros((M, M))

    for i in range(M):
        mean_i = np.sum(X.iloc[:, i]) / N
        for j in range(M):
            mean_j = np.sum(X.iloc[:, j]) / N

            i_data = X.iloc[:,i]
            j_data = X.iloc[:,j]

            # Covariance between column "i" and column "j"
            cov_df_date_after[i, j] = np.sum((i_data - mean_i) * (j_data - mean_j)) / (N - 1)


    df_cov_man = pd.DataFrame(cov_df_date_after)
    df_cov_man.columns = X.columns
    print(f'Manually computed covariance matrix (Problem 5 dataset){nl}{df_cov_man.to_string()}')

    # Problem 11
    cov_auto = np.cov(X, rowvar=False)
    df_cov_auto = pd.DataFrame(cov_auto)
    df_cov_auto.columns = X.columns
    print(f'Automatically computed covariance matrix (Problem 5 dataset){nl}{df_cov_auto.to_string()}')

    if show_plot:
        plt.show()