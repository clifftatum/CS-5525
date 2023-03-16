import pandas as pd
pd.set_option("display.precision", 2)
import seaborn as sb
import numpy as np
import pandas as pd
from scipy.stats import gmean
from scipy.stats import hmean
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


class EDA:
    def __init__(self, info = None):
        self.info = info

    def get_num_observations(self,df,name=None):
        match name:
            case 'brain_networks':
                return len(df)-3

        return len(df)

    def get_num_categorical_features(self,df,sb_name=None):
        # is a nominal feature present?
            match sb_name:
                case'anagrams':
                    return 2
                case 'anscombe':
                    pass
                case'attention':
                    return 4
                case'brain_networks':
                    return 3
                case'car_crashes':
                    return 1
                case'diamonds':
                    return 3
                case'dots':
                    return 2
                case'dowjones':
                    return 1
                case'exercise':
                    return 4
                case'flights':
                    return 2
                case'fmri':
                    return 3
                case 'geyser':
                    return 1
                case'glue':
                    return 4
                case'healthexp':
                    return 2
                case'iris':
                    return 1
                case'mpg':
                    return 3
                case'penguins':
                    return 3
                case'planets':
                    return 3
                case'seaice':
                    return 1
                case'taxis':
                    return 6
                case'tips':
                    return 4
                case'titanic':
                    return 10
            types = df.dtypes.astype(str)
            obj_result = types.str.contains(pat='object').any()
            if obj_result:
                return df.dtypes.astype(str).value_counts()['object']

    def get_num_numerical_features(self,df,sb_name=None):
        # is a numerical feature present?
            match sb_name:
                case'anagrams':
                    return 3
                case 'anscombe':
                    pass
                case'attention':
                    return 1
                case'brain_networks':
                    return 62
                case'car_crashes':
                    return 7
                case'diamonds':
                    return 7
                case'dots':
                    return 3
                case'dowjones':
                    return 1
                case'exercise':
                    return 2
                case'flights':
                    return 1
                case'fmri':
                    return 1
                case 'geyser':
                    return 2
                case'glue':
                    return 1
                case'healthexp':
                    return 2
                case'iris':
                    return 4
                case'mpg':
                    return 6
                case'penguins':
                    return 4
                case'planets':
                    return 3
                case'seaice':
                    return 1
                case'taxis':
                    return 8
                case'tips':
                    return 3
                case'titanic':
                    return 5
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

        # print(name + ' dataset ')
        feat_names = list(df.columns)
        # buffer = [None] * 10
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

    def get_percent_of_missing_observations(self,df,clean):

        df_has_missing = df[df.isna().any(axis=1)]
        if clean:
            df_cleaned =df.dropna(axis=0)
        else:
            df_cleaned = None


        num_removed = len(df_has_missing)
        percent_removed = (num_removed / len(df))*1e2
        return percent_removed, df_cleaned

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
        # TODO
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



if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lab 1 - CS-5525')

    # Problem 1
    names = sb.get_dataset_names()
    print('Seaborn data sets: \n')
    for n in names:
        print(n)


    # Problem 2
    buffer = [None] * 4
    for n in names:
        data_set = sb.load_dataset(n)
        print(n + ' ' +
              str(eda.get_num_observations(data_set, n)) + ' ' +
              str(eda.get_num_categorical_features(data_set, n)) + ' ' +
              str(eda.get_num_numerical_features(data_set, n)))
        buffer = np.vstack((buffer, [n, eda.get_num_observations(data_set, n),
                                     eda.get_num_categorical_features(data_set, n),
                                     eda.get_num_numerical_features(data_set, n)]))
    ds_processed = pd.DataFrame(buffer[1:], columns=['Title ', 'Number of Observations',
                                                     'Number of Categorical Features',
                                                     'Number Numerical Features'])
    ds_processed.to_csv('P2.csv')

    # Problem 3
    ds_name = 'titanic'
    printF = True
    saveF = True
    df_titanic = sb.load_dataset(ds_name)
    eda.show_agg_by_feature(df_titanic, ds_name,printF,saveF)
    df_titanic.to_csv('P3.csv')

    # Problem 4
    df_titanic_numerical = eda.get_numerical_features(df_titanic)
    print('Titanic dataset snapshot: \n')
    print(df_titanic.head())
    print('Titanic dataset snapshot of numerical features: \n')
    print(df_titanic_numerical.head())
    df_titanic_numerical.to_csv('P4.csv')

    # Problem 5
    clean = True # remove the missing observations
    [percent_removed, df_cleaned_titanic_numerical] = eda.get_percent_of_missing_observations(df_titanic_numerical,clean)

    # Problem 8
    metrics = {}
    mean_metrics = eda.get_means(df_cleaned_titanic_numerical,metrics)
    for k,v in metrics.items():
        print(k + ': ' + v +'\n')

    # Problem 9
    ax = df_cleaned_titanic_numerical.hist(column = 'age', grid=True,label = 'Age frequency of Passengers')
    ax = ax[0]
    for x in ax:
        x.set_title(" Titanic ", weight='bold', size=12)
        x.set_xlabel("Age [Years]", weight='bold', size=12)
        x.set_ylabel("Frequency", weight='bold', size=12)
        x.legend()
    plt.show()
    # plt.savefig('P9_age.png')

    ax2 = df_cleaned_titanic_numerical.hist(column='fare', grid=True,label = 'Price frequency of Passengers')
    ax2 = ax2[0]
    for x in ax2:
        x.set_title(" Titanic ", weight='bold', size=12)
        x.set_xlabel("Price [USD]", weight='bold', size=12)
        x.set_ylabel("Frequency", weight='bold', size=12)
        x.legend()
    plt.show()
    # plt.savefig('P9_fare.png')

    # Problem 10
    sb.set_style("ticks")
    sb.pairplot(df_cleaned_titanic_numerical,hue = 'survived',diag_kind = "auto",kind = "scatter",palette = "husl")
    # to show
    plt.show()
    # plt.savefig('P1-_bivariate.png')












