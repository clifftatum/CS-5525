import pandas as pd
import seaborn as sb
import numpy as np

np.random.seed(seed=5525)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format
import numpy as np
import pandas as pd

pd.set_option("display.precision", 2)
from scipy.stats import gmean
from scipy.stats import hmean
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from prettytable import PrettyTable
from DP import data_preprocessing


class EDA:
    def __init__(self, info=None):
        self.info = info

    def get_num_observations(self, df, name=None):
        return len(df)

    def show_feature_datatype(self, df, sb_name=None):
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

    def get_num_categorical_features(self, df, sb_name=None):
        # is a nominal feature present?

        types = df.dtypes.astype(str)
        obj_result = types.str.contains(pat='object').any()
        if obj_result:
            return df.dtypes.astype(str).value_counts()['object']

    def get_num_numerical_features(self, df, sb_name=None):
        # is a numerical feature present?

        types = df.dtypes.astype(str)
        float_result = types.str.contains(pat='float64').any()
        int_result = types.str.contains(pat='int64').any()
        obj_result = types.str.contains(pat='object').any()

        if float_result and int_result:
            return np.sum(
                (df.dtypes.astype(str).value_counts()['int64'], df.dtypes.astype(str).value_counts()['float64']))
        elif float_result and not int_result:
            return df.dtypes.astype(str).value_counts()['float64']
        elif int_result and not float_result:
            return df.dtypes.astype(str).value_counts()['int64']

    def get_numerical_features(self, df, sb_name=None):

        feat_names = list(df.columns)
        num_feat_names = []
        for fn in feat_names:
            feat = df[fn]
            d_type = str(feat.dtypes)
            if d_type == "int64" or d_type == 'float64':
                num_feat_names.append(fn)
        return df[num_feat_names]

    def show_agg_by_feature(self, df, name=None, printF=None, saveF=None):
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
                buffer = np.vstack((buffer, [title, obs_count, mean, std, minimum, quantile_25, quantile_50,
                                             quantile_75, maximum, missing_obs]))
        df_processed = pd.DataFrame(buffer[1:], columns=['Feature Title ', 'Observation Count',
                                                         'Mean', 'STD', 'Min.', '25% Quartile',
                                                         '50% Quartile', '75% Quartile', 'Max.',
                                                         'Missing Observations'])

        return df_processed

    def get_percent_of_missing_observations(self, df, clean, show, clean_method):
        df_cleaned = None
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

                if show:
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
        percent_removed = (num_missing / len(df)) * 1e2
        return num_missing, percent_removed, df_cleaned

    def get_means(self, df, metrics):
        # metrics.update({:})
        feat_names = list(df.columns)

        buffer = [None] * 10
        for fn in feat_names:
            title = str(fn)
            feat = df[fn]
            metrics.update({str(title + ' arithmetic mean'): f' {np.mean(feat):.2f}'})
            metrics.update({str(title + ' geometric mean'): f' {gmean(feat):.2f}'})
            metrics.update({str(title + ' harmonic mean'): f' {hmean(feat):.2f}'})
        return metrics

    def normalize(self, df):
        df_init = self.get_numerical_features(df)
        feat_names = list(df_init.columns)
        df_norm_columns = [f'{df_init.columns[i]} (normalized)' for i in range(0, df_init.shape[1])]
        x_norm_mat = np.zeros((df_init.shape[0], df_init.shape[1]))
        for i, fn in enumerate(feat_names):
            x = df_init[fn]
            x_norm_mat[:, i] = (x - np.min(x)) / (np.max(x) - np.min(x))
        df_norm = pd.DataFrame(x_norm_mat, columns=df_norm_columns)
        return df_norm

    def standardize(self, df):
        df_init = self.get_numerical_features(df)
        feat_names = list(df_init.columns)
        df_stand_columns = [f'{df_init.columns[i]} (standardized)' for i in range(0, df_init.shape[1])]
        x_stand_mat = np.zeros((df_init.shape[0], df_init.shape[1]))
        for i, fn in enumerate(feat_names):
            x = df_init[fn]
            x_stand_mat[:, i] = (x - np.mean(x)) / np.std(x)
        df_stand = pd.DataFrame(x_stand_mat, columns=df_stand_columns)
        return df_stand

    def iq_transform(self, df):
        df_init = self.get_numerical_features(df)
        feat_names = list(df_init.columns)
        df_iqt_columns = [f'{df_init.columns[i]} (iqt)' for i in range(0, df_init.shape[1])]
        x_iqt_mat = np.zeros((df_init.shape[0], df_init.shape[1]))
        for i, fn in enumerate(feat_names):
            x = df_init[fn]
            x_iqt_mat[:, i] = (x - x.quantile(.5)) / (x.quantile(.75) - x.quantile(.25))
        df_iqt = pd.DataFrame(x_iqt_mat, columns=df_iqt_columns)
        return df_iqt

    def minkowski_distance(self, x, y, L):
        m_dist = np.zeros((len(x), 1))
        for i in range(0, len(x)):
            sum_abs_raised = sum(abs(np.array([x[i], y[i]])) ** L)
            m_dist[i] = sum_abs_raised ** (1 / L)
        return m_dist.reshape(len(m_dist), 1)

    def plot_all_features(self, df, x_feature, df_name, on_subplots, plot_on_this):
        if x_feature is not None:
            df['Date'] = x_feature
        if len(df.shape) > 1:
            n_plots = df.shape[1]
            n_col = int(n_plots / 2)
            if n_plots % 2 == 0:
                n_row = (n_plots - n_col)
            else:
                n_row = (n_plots - n_col) + 1
            left = n_row * n_col - n_plots
            if left > 0 and left > n_col:
                n_row = n_row - 1

        else:
            n_plots = 1
            n_row = 1
            n_col = 1

        if plot_on_this is not None:
            df.plot(x='Date', kind='line', subplots=on_subplots, grid=True, title=df_name,
                    layout=(n_row, n_col), sharex=False, sharey=False, legend=True, ax=plot_on_this)
        else:
            df.plot(x='Date', kind='line', subplots=on_subplots, grid=True, title=df_name,
                    layout=(n_row, n_col), sharex=False, sharey=False, legend=True)
            plot_on_this = plt.gcf().axes

        for i, ax in enumerate(plt.gcf().axes):
            ax.set_xlabel('Observations')
            try:
                ax.set_ylabel(df.columns[i])
            except:
                pass
        plt.tight_layout()
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.01, hspace=.01)

        if x_feature is not None:
            n_samples = len(x_feature)
            plt.xticks(np.array([x_feature.iloc[0],
                                 x_feature.iloc[int(1 * (n_samples / 6))],
                                 x_feature.iloc[int(2 * (n_samples / 6))],
                                 x_feature.iloc[int(3 * (n_samples / 6))],
                                 x_feature.iloc[int(4 * (n_samples / 6))],
                                 x_feature.iloc[int(5 * (n_samples / 6))],
                                 x_feature.iloc[int(6 * (n_samples / 6)) - 1]
                                 ]))

        return plot_on_this

    def get_pca(self, df):
        pca = PCA(n_components=len(df.columns), svd_solver='full')
        pca.fit(df)
        X_PCA = pca.transform(df)
        return X_PCA

    def plot_features_unified_xaxis(self, df, x_axis_feature, y_axis_feature, observation_ID, show_plot, title):
        plt.figure(figsize=(12, 8))
        legFlag = True
        for obs in observation_ID:
            for i, yy in enumerate(y_axis_feature):
                x = df[df['symbol'] == obs][x_axis_feature]
                y = df[df['symbol'] == obs][y_axis_feature]
                if y.isnull().values.any():
                    data = df.dropna()
                    x = data[data['symbol'] == obs][x_axis_feature]
                    y = data[data['symbol'] == obs][y_axis_feature]
                if title == 'Google stock price and moving average' and i == 1:
                    plt.plot(x, y, label=yy)
                elif title == 'Google stock price and moving average':
                    if legFlag:
                        plt.plot(x, y, label=obs)
                        legFlag = False
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
                             x.iloc[int(1 * (n_samples / 6))],
                             x.iloc[int(2 * (n_samples / 6))],
                             x.iloc[int(3 * (n_samples / 6))],
                             x.iloc[int(4 * (n_samples / 6))],
                             x.iloc[int(5 * (n_samples / 6))],
                             x.iloc[int(6 * (n_samples / 6)) - 1]
                             ]))

    def slice_by_observation(self, df, feature, observations):
        cmmd_str = ''
        temp = np.zeros((df.shape[0], 1))
        df_temp = pd.DataFrame(temp)
        if observations is None:
            for i, feat in enumerate(feature):
                cmmd_str = ' '"'" + feat + "'"' '
                exec('df_temp[' + cmmd_str + '] =  df[' + cmmd_str + ']')
            return df_temp.iloc[:, 1:]
        else:
            for i, obs in enumerate(observations):
                if i != len(observations) - 1:
                    cmmd_str = cmmd_str + '(df['"'" + feature + "'"'] == '"'" + obs + "'" ') | '
                else:
                    cmmd_str = cmmd_str + '(df['"'" + feature + "'"'] == ' "'" + obs + "'" ')'

        return eval('df[' + cmmd_str + ']')

    def compute_skewness(self, data, order):
        N = len(data)
        m_i = (1 / N) * np.sum((data - np.mean(data)) ** order)
        return m_i

    def to_pretty_table(self, dat, title, head):
        pd.set_option("display.precision", 2)
        if isinstance(dat, pd.DataFrame):
            data = dat.round(decimals=2).values
            if head is None:
                headers = dat.columns
        else:
            data = np.round(dat, 2)
            headers = list(head)

        x = PrettyTable(data.dtype.names)
        for row in data:
            x.add_row(row)
        x.field_names = list(headers)
        if title is not None:
            x.title = title
        print(x)

    def plot_data_with_principal_axes(self, df, ds_title):
        data = df.values
        mu = data.mean(axis=0)
        data = data - mu
        cov = np.cov(df, rowvar=False)
        eigenvalues,eigenvectors = np.linalg.eig(cov)
        projected_data = np.dot(data, eigenvectors)
        sigma = projected_data.std(axis=0).mean()

        fig, ax = plt.subplots()
        colors = ['red', 'blue']
        plt.scatter(data[:, 0], data[:, 1])
        norm_eigen = normalize(eigenvalues.reshape(1, len(eigenvalues))) + (
                    1 - np.max(normalize(eigenvalues.reshape(1, len(eigenvalues)))))
        for i, axis in enumerate(eigenvectors):
            end = 0 +(sigma * np.multiply(axis, *norm_eigen[:,i]))
            end[0] = end[0]*-1
            start = 0,0
            plt.scatter(0, 0, color=colors[i],
                        label='Feature: ' + list(df.columns)[i] + ' - Eigenvalue = ' + str(round(eigenvalues[i], 2)),
                        marker='<', s=1)

            ax.annotate(
                '', xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(facecolor=colors[i], width=2.0))
        ax.set_aspect('equal')

        plt.xlabel(df.columns[0] + 'rotated into Eigenspace')
        plt.ylabel(df.columns[1] + 'rotated into Eigenspace')
        plt.title(ds_title)
        plt.legend()

    def time_difference(self, t, order):
        diff = list()
        for i in range(order, len(t)):
            value = t[i] - t[i - order]
        diff.append(value)
        return np.array(diff).reshape(len(diff), 1)


if __name__ == '__main__':
    eda = EDA('Clifford Tatum HW 2 - CS-5525')
    dp = data_preprocessing()
    plot = True
    save = False

    start_date = "2000-01-1"
    end_date = "2022-09-25"
    stock_ticker = "AAPL"
    start = pd.to_datetime([start_date]).astype(int)[0] // 10 ** 9  # convert to unix timestamp.
    end = pd.to_datetime([end_date]).astype(int)[0] // 10 ** 9  # convert to unix timestamp.
    url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock_ticker + '?period1=' + str(
        start) + '&period2=' + str(end) + '&interval=1d&events=history'
    df_APPLE = pd.read_csv(url)

    # Problem 1
    df_APPLE_norm = dp.normalize(df_APPLE)
    df_APPLE_stand = dp.standardize(df_APPLE)
    df_APPLE_iqt = dp.iq_transform(df_APPLE)

    x_axis = pd.to_datetime(df_APPLE['Date'])
    title1 = "AAPL Stock " + str(start_date) + " to " + str(end_date) + "  - Raw data"
    title2 = "AAPL Stock " + str(start_date) + " to " + str(end_date) + "  - Normalized data"
    title3 = "AAPL Stock " + str(start_date) + " to " + str(end_date) + "  - Standardized data"
    title4 = "AAPL Stock " + str(start_date) + " to " + str(end_date) + "  - Inter-Quartile Normalized data"
    fig, axs = plt.subplots(2, 2)
    ax1 = dp.plot_all_features(df=df_APPLE,
                               x_feature=x_axis,
                               df_name=title1,
                               on_subplots=False,
                               plot_on_this=axs[0, 0])
    ax2 = dp.plot_all_features(df=df_APPLE_norm,
                               x_feature=x_axis,
                               df_name=title2,
                               on_subplots=False,
                               plot_on_this=axs[0, 1])
    ax3 = dp.plot_all_features(df=df_APPLE_stand,
                               x_feature=x_axis,
                               df_name=title3,
                               on_subplots=False,
                               plot_on_this=axs[1, 0])
    ax4 = dp.plot_all_features(df=df_APPLE_iqt,
                               x_feature=x_axis,
                               df_name=title4,
                               on_subplots=False,
                               plot_on_this=axs[1, 1])

    fig1 = plt.gcf()
    fig1.set_size_inches(14, 12)
    # Problem 3
    Ls = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
    n_samps = 50000
    Y = np.random.standard_normal(n_samps, )
    X = np.random.standard_normal(n_samps, )
    mink_dist_mat = np.zeros((len(X), len(Ls)))
    plt.figure()
    for i, L in enumerate(Ls):
        mink_dist_mat[:, i] = eda.minkowski_distance(x=X, y=Y, L=L).flatten()
        p_norm_inv = np.divide(1, mink_dist_mat[:, i]).reshape(n_samps, 1)
        plot_mink_x = np.multiply(X.reshape(n_samps, 1), p_norm_inv)
        plot_mink_y = np.multiply(Y.reshape(n_samps, 1), p_norm_inv)
        plt.scatter(plot_mink_x, plot_mink_y, label='L' + str(L) + ' norm', s=.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lr norm')
    plt.legend()
    plt.legend(loc="upper right")

    df_iqt_columns = [f'L{i} norm' for i in Ls]
    df_mink_dist = pd.DataFrame(mink_dist_mat, columns=df_iqt_columns)
    fig2 = plt.gcf()
    fig2.set_size_inches(14, 12)

    # Problem 6
    mean = 1
    var = 2
    N = 1000
    x = np.random.normal(mean, var, N)

    mean = 2
    var = 3
    N = 1000
    epsilon = np.random.normal(mean, var, N)

    y = x + epsilon

    # Covariance matrix for X against Y
    X = pd.DataFrame(np.hstack((x.reshape(len(x), 1), y.reshape(len(y), 1))), columns=['X', 'Y'])

    M = X.shape[1]
    N = eda.get_num_observations(X)
    covariance = np.zeros((M, M))

    for i in range(M):
        mean_i = np.sum(X.iloc[:, i]) / N
        for j in range(M):
            mean_j = np.sum(X.iloc[:, j]) / N

            i_data = X.iloc[:, i]
            j_data = X.iloc[:, j]

            # Covariance between column "i" and column "j"
            covariance[i, j] = np.sum((i_data - mean_i) * (j_data - mean_j)) / (N - 1)

    df_cov = pd.DataFrame(covariance, columns=['X covariance (ith vs jth)', 'Y covariance (ith vs jth)'])
    eda.to_pretty_table(df_cov, title='Covariance Matrix of Feature Matrix', head=None)

    E_val,E_vec = np.linalg.eig(covariance)
    df_eig = pd.DataFrame(E_vec,
                          columns=[f'Eigenvector for Eigenvalue = {round(E_val[i], 2)}' for i in range(0, len(E_val))])

    eda.to_pretty_table(df_eig, title='Eigenvalue & Eigenvector', head=None)
    eda.plot_data_with_principal_axes(X, ds_title='Random Data X and Y')
    fig3 = plt.gcf()
    fig3.set_size_inches(14, 12)

    E_vec, E_val, singular_values = np.linalg.svd(X.values, full_matrices=False)
    eda.to_pretty_table(singular_values, title='Singular Values of Feature Matrix', head=list(X.columns))

    corr = X.corr()
    eda.to_pretty_table(corr, title='Correlation Matrix of Feature Matrix', head=None)

    # Problem 7

    # create a differenced series
    x_t = np.arange(-4, 5, 1)
    y_t = x_t ** 3
    df_ts = pd.DataFrame(np.hstack((x_t.reshape(len(x_t), 1), y_t.reshape(len(y_t), 1))))
    df_ts.columns = ['x_t', 'y_t']
    order = 1

    df_ts['delta 1 y_t'] = df_ts['y_t'].diff(periods=1)
    df_ts['delta 2 y_t'] = df_ts['y_t'].diff(periods=2)
    df_ts['delta 3 y_t'] = df_ts['y_t'].diff(periods=3)

    eda.to_pretty_table(df_ts,
                        title='1st, 2nd, and 3rd Order time differencing for cubic function',
                        head=None)

    df_y = eda.slice_by_observation(df_ts,
                                    feature=['y_t', 'delta 1 y_t', 'delta 2 y_t', 'delta 3 y_t'],
                                    observations=None)

    eda.plot_all_features(df=df_y,
                          x_feature=df_ts['x_t'],
                          df_name='1st, 2nd, and 3rd Order time differencing for cubic function',
                          on_subplots=False,
                          plot_on_this=None)
    fig4 = plt.gcf()
    fig4.set_size_inches(14, 12)

    if plot:
        if save:
            import pylab as pl

            pl.figure(fig1)
            plt.savefig("P2.png", dpi=100)
            pl.figure(fig2)
            plt.savefig("P3.png", dpi=100)
            pl.figure(fig3)
            plt.savefig("P6.png", dpi=100)
            pl.figure(fig4)
            plt.savefig("P7.png", dpi=100)
        plt.show()
