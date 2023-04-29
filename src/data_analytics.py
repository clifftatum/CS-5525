import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import tree
from scipy.stats import gmean
from scipy.stats import hmean
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import chi2_contingency, f_oneway
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from scipy import stats

np.random.seed(5525)
pd.set_option("display.precision", 3)


class ExploratoryDataAnalysis:
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

        types = df.dtypes.type(str)
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

    def get_aggregate(self, df):
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

    def show_aggregate(self, df, df_agg, plot_type, title,split):
        if plot_type == 'plotly':
            if df_agg is not None:
                n_rows = 2
                n_cols = 1
                specs = [[{"type": "table"}],
                       [{"type": "xy"}]]
                # Create the figure
                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    specs=specs
                )
            else:
                n_rows = 1
                n_cols = 1
                if split is not None:
                    n_cols = 2
                # Create the figure
                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    shared_xaxes=True,
                    vertical_spacing=0.03
                )



            if df_agg is not None:
                # add the Aggregate Table
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=df_agg.columns,
                            font=dict(size=10),
                            align="left"
                        ),
                        cells=dict(
                            values=[df_agg[k].tolist() for k in df_agg.columns],
                            align="left")
                    ),
                    row=1, col=1
                )


            df_numeric = self.get_numerical_features(df)
            for i, feat in enumerate(df_numeric.columns):
                if feat in split:
                    fig.add_trace(go.Violin(points=False,y=df_numeric[feat],
                                            box_visible=True,
                                            meanline_visible=True,
                                            name=feat),
                                  row=n_rows, col=n_cols)
                else:
                    fig.add_trace(go.Violin(points=False,y=df_numeric[feat],
                                            box_visible=True,
                                            meanline_visible=True,
                                            name=feat),
                                  row=n_rows, col=n_cols-1)

            fig.update_yaxes(type="log", row=2, col=1)
            fig.update_yaxes(tickfont_family="Arial Black")
            fig.update_xaxes(tickfont_family="Arial Black")
            fig.update_layout(template="plotly_dark",font=dict(size=18))
            fig.update_layout(
                showlegend=True,
                title_text=title,
            )
            return fig

    def get_percent_of_missing_observations(self, df, clean, show, clean_method):
        df_cleaned = None
        df_has_missing = df[df.isna().any(axis=1)]
        if show:
            print('The number of missing entries by feature are \n')
            print(f'{df.isna().sum()}')

            print(f'The total number of '
                  f'missing entries are {df.isna().sum().sum()}')

        if clean:
            if clean_method == None or clean_method == 'mean':
                df_cleaned = df.fillna(df.mean(axis=0, skipna=True))
            elif clean_method == 'prune':
                dirty_feats = str(df.columns[df.isna().any()][0])
                df = df.drop(columns=dirty_feats)
                df_cleaned = df
            elif clean_method == 'zeros':
                df_cleaned = df.fillna(0)

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

    def detect_and_remove_outliers(self,df,method ='LOF'):

        # Extract features
        features = df.columns
        X = df
        if method == 'LOF':
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            y_hat = lof.fit_predict(X)
        elif method == '1ClassSVM':
            y_hat = OneClassSVM(nu=0.01).fit_predict(X)


        keep_ind = np.where(y_hat == 1)[0]
        return keep_ind, (1-(len(keep_ind) / len(df))) * 100
        # filter outliers
        # X_filtered = X[y_hat == 1]
        # Create scatter plot with outliers highlighted
        # fig = px.scatter(df, x='feature1', y='feature2', color=y_hat, title='Outlier Detection')
        # fig.show()
        #
        # # Create scatter plot of filtered data
        # fig_filtered = px.scatter(pd.DataFrame(X_filtered, columns=features), x='feature1', y='feature2', title='Filtered Data')
        # fig_filtered.show()
        #
        # percent_detected_and_removed =(1-(len(df[filtered_entries])/len(df)))*100
        # return df_father[filtered_entries], percent_detected_and_removed



    def standardize(self, df):
        from sklearn.preprocessing import StandardScaler
        df_stand = StandardScaler().fit_transform(df)
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

    def test_null_hypothosis(self,X,y):

        # # Kai Squared
        # observed_freq = np.array([[np.sum((X == i) & (y == j)) for j in np.unique(y)] for i in np.unique(X)])
        # chi2_statistic, p_value, dof, expected_freq = chi2_contingency(observed_freq)
        #
        # print("Chi-squared test results:")
        # print("Chi-squared statistic = {:.3f}".format(chi2_statistic))
        # print("p-value = {:.3f}".format(p_value))

        # Perform ANOVA test
        groups = [X[y == g] for g in np.unique(y)]
        f_statistic, p_value = f_oneway(*groups)

        # print("ANOVA test results:")
        # print(f"F-statistic = {f_statistic}")
        # print(f"p-value = {p_value}")

        df_res = pd.DataFrame(np.hstack((np.array(X.columns).reshape(-1,1),f_statistic.reshape(-1,1),p_value.reshape(-1,1))),
                              columns = ['Feature','F Statistic','P-value'])

        ExploratoryDataAnalysis.to_pretty_table(self,dat=df_res, title='ANOVA Null Hypothesis Test Results', head=None)

        return df_res

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
                    sharex=False, sharey=False, legend=True)

            # df.plot(subplots=on_subplots)
            # plt.tight_layout()
            plot_on_this = plt.gcf().axes

        for i, ax in enumerate(plt.gcf().axes):
            if i == len(plt.gcf().axes) - 1:
                ax.set_xlabel('Observations')
            try:
                # ax.set_ylabel(df.columns[i])
                pass
            except:
                pass

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
        # plt.tight_layout()
        return plot_on_this

    def get_pca(self, df, show_cum_exp_var, required_exp_variance):
        pca = PCA(n_components=df.shape[1])
        the_PCA = pca.fit(df)
        cum_exp_var = np.cumsum(the_PCA.explained_variance_ratio_)
        n_required_features = np.argwhere(cum_exp_var > required_exp_variance)[0] + 1
        if show_cum_exp_var:
            fig = px.area(x=np.arange(1, df.shape[1] + 1), y=cum_exp_var)
            fig.update_xaxes(showgrid=True, tickvals=np.arange(1, df.shape[1] + 1), )
            fig.update_yaxes(showgrid=False)
            fig.update_yaxes(tickfont_family="Arial Black")
            fig.update_xaxes(tickfont_family="Arial Black")
            fig.update_layout(template="plotly_dark",font=dict(size=18))
            fig.update_layout(showlegend=True,
                              yaxis_range=[0, 1.02],
                              xaxis_title='<b>Number of Features<b>',
                              yaxis_title='<b>Cumulative Explained Variance<b>',
                              title_text='<b>Principle Component Analysis (PCA)<b>')

        else:
            fig = None

        return n_required_features, fig

    def show_cov(self,df):
        # cov_auto = np.cov(df, rowvar=False)
        # df_cov_auto = pd.DataFrame(cov_auto)
        # df_cov_auto.columns = df.columns
        cov_auto =df.cov()

        # Plot Covariance matrix
        fig = go.Figure()
        fig.add_trace((go.Heatmap( x=cov_auto.columns,
                                   y=cov_auto.index,
                                   z = np.flip(np.array(cov_auto),0),
                                   colorscale='viridis')))
        fig.update_layout(title_text='<b>Covariance Matrix<b>')
        fig.update_yaxes(tickfont_family="Arial Black")
        fig.update_xaxes(tickfont_family="Arial Black")
        fig.update_layout(template="plotly_dark")
        fig.update_yaxes(scaleanchor="x",scaleratio=1)
        df_cov_auto = pd.DataFrame(cov_auto)
        nl = '\n'
        print(f'Covariance matrix: {nl}{df_cov_auto.to_string()}')

        return fig

    def show_corr(self,df):
        corr_mat = df.corr()
        df_corr = pd.DataFrame(corr_mat)
        df_corr.columns = df.columns

        # Plot Correlation matrix
        fig = go.Figure()
        fig.add_trace((go.Heatmap(x=df.columns,
                                  y=df.columns,
                                  z = np.flip(np.array(df_corr),0),
                                  colorscale='viridis')))
        fig.update_layout(title_text='<b>Pearson Correlation Coefficient Matrix<b>')
        fig.update_yaxes(tickfont_family="Arial Black")
        fig.update_xaxes(tickfont_family="Arial Black")
        fig.update_layout(template="plotly_dark")
        fig.update_yaxes(scaleanchor="x",scaleratio=1)

        df_corr = pd.DataFrame(corr_mat)
        nl = '\n'
        print(f'Pearson Correlation Coefficient Matrix: {nl}{df_corr.to_string()}')

        return fig




    def get_svd(self, df, show_cum_exp_var, required_exp_variance):

        # perform SVD on the features
        U, S, Vt = np.linalg.svd(df)

        # plot the explained variance ratio
        explained_var = (S ** 2) / (S ** 2).sum()

        cum_exp_var = np.cumsum(explained_var)
        n_required_features = np.argwhere(cum_exp_var > required_exp_variance)[0] + 1
        if show_cum_exp_var:
            fig = px.area(x=np.arange(1, df.shape[1] + 1), y=cum_exp_var)
            fig.update_xaxes(showgrid=True, tickvals=np.arange(1, df.shape[1] + 1), )
            fig.update_yaxes(showgrid=False)
            fig.update_yaxes(tickfont_family="Arial Black")
            fig.update_xaxes(tickfont_family="Arial Black")
            fig.update_layout(template="plotly_dark",font=dict(size=18))
            fig.update_layout(showlegend=True,
                              yaxis_range=[0, 1.02],
                              xaxis_title='<b>Number of Features<b>',
                              yaxis_title='<b>Cumulative Explained Variance<b>',
                              title_text='<b>Singular Value Decomposition (SVD) Analysis<b>')


        else:
            fig = None

        return n_required_features, fig

    def plot_features_unified_xaxis(self, df, x_axis_feature, y_axis_feature, observation_ID, plot_type, title):
        plt.figure(figsize=(12, 8))
        legFlag = True
        for obs in observation_ID:
            for i, yy in enumerate(y_axis_feature):
                data = df.dropna()
                x = data[data['Symbol'] == obs][x_axis_feature]
                y = data[data['Symbol'] == obs][y_axis_feature]
                if plot_type == 'matplotlib':
                    plt.plot(x, y, label=obs)
                    fig = plt.gcf()
                elif plot_type == 'plotly':
                    fig = px.plot(x, y, label=obs)

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

    def slice_by_observation(self, df, feature, observations, obs_by_feature):
        temp = np.zeros((df.shape[0], 1))
        df_temp = pd.DataFrame(temp)
        if observations is not None:
            obs = observations[0]
            obs_by_feat = obs_by_feature[0]
        for i, feat in enumerate(feature):
            cmmd_str = ' '"'" + feat + "'"' '
            if observations is None:
                exec('df_temp[' + cmmd_str + '] =  df[' + cmmd_str + ']')
            else:
                for obs in observations:
                    # df['Symbol'].loc[df['Symbol'] == 'A']
                    exec(
                        'df_temp[' + cmmd_str + '] =  df[' + cmmd_str + '].loc[df['"'" + obs_by_feat + "'" '] == '"'" + obs + "'" ']')
        df_send = df_temp.iloc[:, 1:]
        df_send_trunc = df_send.dropna()
        return df_send_trunc

    def compute_skewness(self, data, order):
        N = len(data)
        m_i = (1 / N) * np.sum((data - np.mean(data)) ** order)
        return m_i

    def to_pretty_table(self, dat, title, head):
        pd.set_option("display.precision", 3)
        if isinstance(dat, pd.DataFrame):
            data = dat.round(decimals=3).values
            if head is None:
                headers = dat.columns
        else:
            try:
                data = np.round(dat, 3)
                headers = list(head)
            except:
                data = dat
                headers = ''

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
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        projected_data = np.dot(data, eigenvectors)
        sigma = projected_data.std(axis=0).mean()

        fig, ax = plt.subplots()
        colors = ['red', 'blue']
        plt.scatter(data[:, 0], data[:, 1])
        norm_eigen = normalize(eigenvalues.reshape(1, len(eigenvalues))) + (
                1 - np.max(normalize(eigenvalues.reshape(1, len(eigenvalues)))))
        for i, axis in enumerate(eigenvectors):
            end = 0 + (sigma * np.multiply(axis, *norm_eigen[:, i]))
            end[0] = end[0] * -1

            start = 0, 0

            # fig.add_trace(go.Scatter(name="CI", x=rec_res.n_observations,
            #                          y=df_predictions.obs_ci_lower.iloc[:],
            #                          line=dict(color='rgba(0, 0, 255, 0.05)'),
            #                          fillcolor="#eaecee",
            #                          connectgaps=False,
            #                          ))

            plt.scatter(start[0], start[1], color=colors[i],
                        label='Feature: ' + df.columns[i] + ' - Eigenvalue = ' + str(round(eigenvalues[i], 2)),
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

    def random_forest_analysis(self, X, y, plot_type, title, max_features):

        model = RandomForestRegressor(random_state=1,
                                      max_depth=5,
                                      n_estimators=100)

        types = pd.DataFrame(y).dtypes.astype(str)
        obj_result = types.str.contains(pat='object').any()
        if obj_result:
            # y = pd.DataFrame(y)
            # y['rank'] = None
            # for index, row in y.iterrows():
            #     y['rank'][index] = rank_list.index(row['target_buy_sell_performance'])
            # y = y['rank']
            # Label encode categorical response variable
            le = LabelEncoder()
            y = le.fit_transform(y)

        model.fit(X, y)

        features = X.columns
        importance = model.feature_importances_
        indices = np.argsort(importance) # show all

        keep_features = [features[i] for i in indices]
        remove_features = list(set(features) - set(keep_features))

        indices = np.argsort(importance)[:max_features]  # show all
        keep_features = [features[i] for i in indices]
        would_remove_features = list(set(features) - set(keep_features))

        # Show the dropped features
        # self.to_pretty_table(dat = dropped_feats,title='Dropped Features:',head='None')
        print(f'Dropped Features: {would_remove_features}')

        # Show the kept features
        # self.to_pretty_table(dat = x_train_init.columns,title='Kept Features:',head='None')
        print(f'Kept Features: {keep_features}')

        if plot_type == 'matplotlib':
            plt.barh(range(len(indices)), importance[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            fig = plt.gcf()
        elif plot_type == 'plotly':
            fig = px.bar(range(len(indices)), importance[indices], orientation='h')

            fig.update_layout(xaxis_title="Relative Importance",
                              yaxis=dict(
                                  tickvals=np.arange(0, len(indices)),
                                  ticktext=[features[i] for i in indices]),
                              showlegend=True,
                              title_text=title,
                              )
            fig.update_yaxes(tickfont_family="Arial Black")
            fig.update_xaxes(tickfont_family="Arial Black")
            fig.update_layout(template="plotly_dark",font=dict(size=12))

            return fig, would_remove_features

    def show_hbar(self, df, x_feat, y_feat, by_leg_category):
        fig = px.bar(df, x=x_feat, y=y_feat, color=by_leg_category, orientation='h')

        fig.update_layout(xaxis_title=x_feat, yaxis_title=y_feat,
                          title_text=x_feat + ' by ' + y_feat,
                          )
        fig.update_layout(
            plot_bgcolor='white'
        )
        fig.update_xaxes(
            mirror=False,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )

        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)

        return fig

    def one_hot_encode(self, df,not_features):
        init_len = len(df.columns)
        ind_to_standardize = []
        encoder = OneHotEncoder(handle_unknown='ignore')
        df_quant_feat = self.get_numerical_features(df).columns
        df_all = df.columns
        df_qual_feat = [ele for ele in df_all if ele not in df_quant_feat ]
        df_qual_feat = [l for l in df_qual_feat if l not in not_features ]
        final_df = df
        count_new_columns = 0

        encoded_feat_names = []
        for qf in df_qual_feat:

            # perform one-hot encoding on each qualitative column
            encoder_df = pd.DataFrame(encoder.fit_transform(df[[qf]]).toarray())

            temp_df = pd.DataFrame(np.hstack((df[qf].values.reshape(len(df), 1), encoder_df.values)))
            # for u_cat in final_df.iloc[:,0].unique():
            col_names = [qf]
            columns = df[qf].unique()

            for cat in columns:
                encoded_feat_names.append((qf + '_' + cat))
                final_df[(qf + '_' + cat)] = [1 if i==cat else 0 for i in final_df[qf] ]
            final_df.drop(qf, axis=1, inplace=True)

        ind_to_standardize = np.arange(len(final_df.columns) - (count_new_columns),
                                       len(final_df.columns) - (count_new_columns) + count_new_columns)
        return final_df, ind_to_standardize,encoded_feat_names

    def split_80_20(self, df, target):
        y = pd.DataFrame(df[target], columns=[target])
        df.drop(columns=target, inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(df, y,
                                                            test_size=0.2,
                                                            random_state=1,
                                                            shuffle=False)  # False if using temporal data
        return x_train, x_test, y_train, y_test


class RegressionAnalysis:
    def __init__(self, info=None):
        pass

    @staticmethod
    def drop_and_show_regression_results(x_train, y_train, x_test, y_test, dropped_feats, show, title,
                                     compute_prediction,
                                     compute_method, dim_red_method):
        # Drop the features
        if dropped_feats is not None:
            x_train.drop(columns=dropped_feats, axis=1, inplace=True)

        types = pd.DataFrame(y_train).dtypes.astype(str)
        obj_result = types.str.contains(pat='object').any()
        if obj_result:
            # Label encode categorical response variable
            le = LabelEncoder()
            y_train[y_train.columns[0]] = le.fit_transform(y_train)
            y_test[y_train.columns[0]] = le.fit_transform(y_test)

        # Compute the linear regression
        model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
        # logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        # model = logreg.fit(x_train, y_train)
        fig = None
        if show:
            print(f'{dim_red_method} Model Summary: {model.summary2(float_format="%.3f")}')
            if compute_prediction and compute_method == 'package':
                # Compute a prediction on the test data set
                # Using Package
                Y = y_test
                if dropped_feats is not None:
                    x_test = x_test.drop(columns=dropped_feats)
                # X = sm.add_constant(x_test)
                y_hat_model = sm.OLS(Y, x_test).fit()
                # y_hat_model = model
                y_hat = y_hat_model.predict(x_test)
                # Decode predicted labels
                # y_hat = le.inverse_transform(y_hat)
                MSE = mean_squared_error(np.array(Y), np.array(y_hat).reshape(len(y_hat), 1))

            # print(f' The mean squared error of the OLS model against the test data is {MSE:.3f}')
            # Plot
            x_test[Y.columns[0]] = Y
            x_test['Observations'] = np.arange(1, len(x_test) + 1).reshape(len(x_test), 1)
            x_test['OLS_predicted_' + Y.columns[0]] = np.array(y_hat).reshape(len(y_hat), 1)
            x_test['Predicted ' + Y.columns[0]] = np.array(y_hat).reshape(len(y_hat), 1)

            fig = px.line(x_test, x_test['Observations'], y=[Y.columns[0], 'Predicted ' + Y.columns[0]])

            fig.update_layout(plot_bgcolor='white')
            fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
            fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=False)
            if title == None:
                title_str = dim_red_method + ' Resultant Feature Space:  Predicted ' \
                            + Y.columns[0] + ' vs Actual ' + \
                            Y.columns[0] + ': MSE = ' + str(MSE)
            else:
                title_str = title

            fig.update_layout(showlegend=True,
                              title_text=title_str,
                              yaxis_title=Y.columns[0])

        return y_hat_model, x_test, fig

    def backward_linear_regression(self, x_train, y_train, x_test, y_test, compute_prediction, compute_method, show,
                                   encode_target):

        types = pd.DataFrame(y_train).dtypes.astype(str)
        obj_result = types.str.contains(pat='object').any()
        if encode_target and obj_result:
            # Label encode categorical response variable
            le = LabelEncoder()
            y_train[y_train.columns[0]] = le.fit_transform(y_train)
            y_test[y_train.columns[0]] = le.fit_transform(y_test)

        x_train_init = x_train.copy(deep=True)
        models = {}
        prediction = None
        dropped_feats = []
        R_squared_rates = []
        adjRsqr = []
        potential_best_model = None
        for i, feat in enumerate(x_train):
            if i == 0:
                msg_str = 'full feature space'
                prev_adj_R_squared = None
            else:
                prev_adj_R_squared = model.rsquared_adj

            # Compute the linear regression
            model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
            if i == 0:
                print(f'Initial model: {model.summary2(float_format="%.3f")}')

            # Measure the model, the R_squared and the Adjusted R squared, the feature with the max p-value
            adjRsqr.append(model.rsquared_adj)

            # Track the change in adj Rsq by pruned features
            if i > 0:
                R_squared_rates.append(np.round((model.rsquared_adj - prev_adj_R_squared) * 100, 2))
            else:
                R_squared_rates.append(0)

            models.update({prev_adj_R_squared: model})
            max_p_ind = np.argmax(model.pvalues.iloc[1:])
            if np.max(model.pvalues.iloc[1:]) <= 0.05:  # if the max p-value is below the acceptable threshold , break
                temp = x_train_init.drop(columns=dropped_feats)
                potential_best_model = sm.OLS(y_train, temp).fit()
                break
            # Drop the feature with the highest p value
            dropped_feats.append(x_train.columns[max_p_ind])
            prev_msg_str = msg_str
            msg_str = 'Dropped: ' + str(dropped_feats)
            x_train.drop(columns=x_train.columns[max_p_ind], inplace=True)

        # Drop the features
        x_train_init.drop(columns=dropped_feats, inplace=True)
        if compute_prediction:
            # Show the dropped features
            # self.to_pretty_table(dat = dropped_feats,title='Dropped Features:',head='None')
            print(f'Dropped Features: {dropped_feats}')

            # Show the kept features
            # self.to_pretty_table(dat = x_train_init.columns,title='Kept Features:',head='None')
            print(f'Kept Features: {x_train_init.columns}')

            # Show the best model summary
            min_adjusted_R_sqrd_rate = min(R_squared_rates)
            if min_adjusted_R_sqrd_rate > -5:
                print(f'Best Model Summary: {potential_best_model.summary2(float_format="%.3f")}')
            else:
                ind_select_model = np.argmin(R_squared_rates) - 1
                potential_best_model = models[adjRsqr[ind_select_model]]
                dropped_feats = dropped_feats[0:ind_select_model]
                print(f'Best Model Summary: {potential_best_model.summary2(float_format="%.3f")}')

            # Compute a prediction on the test data set
            if compute_prediction and compute_method == 'manual':
                # Manual
                x_test = x_test.drop(columns=dropped_feats)
                # X = np.hstack((np.ones(len(x_test)).reshape(len(x_test), 1), x_test))
                X = sm.add_constant(x_test)
                Y = y_test
                H = X.T @ X
                Beta = np.linalg.inv(H) @ X.T @ Y
                y_hat = X @ Beta  # here is B* or the predicition
                e = np.array(Y) - np.array(y_hat)
                SSE = e.T @ e
                MSE = np.mean(e ** 2)

            elif compute_prediction and compute_method == 'package':
                # Using Package
                Y = y_test
                x_test = x_test.drop(columns=dropped_feats)
                # X = np.hstack((np.ones(len(x_test)).reshape(len(x_test), 1), x_test)) \
                X = sm.add_constant(x_test)
                y_hat_model = model # .predict(exog=X)
                y_hat = y_hat_model.predict(X)
                e = np.array(Y) - np.array(np.array(y_hat).reshape(len(y_hat), 1))
                SSE = e.T @ e

                MSE = mean_squared_error(Y, np.array(y_hat).reshape(-1, 1))
            if show:
                # print(f' The mean squared error of the OLS model against the test data is {MSE:.3f}')

                x_test[Y.columns[0]] = Y
                x_test['n_observations'] = np.arange(1, len(x_test) + 1).reshape(len(x_test), 1)
                x_test['OLS_predicted_' + Y.columns[0]] = np.array(y_hat).reshape(len(y_hat), 1)
                x_test['Predicted ' + Y.columns[0]] = np.array(y_hat).reshape(len(y_hat), 1)
                fig = px.line(x_test, x_test['n_observations'], y=[Y.columns[0], 'Predicted ' + Y.columns[0]])

                fig.update_layout(plot_bgcolor='white')
                fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
                fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=False)
                fig.update_layout(showlegend=True,
                                  yaxis_title=Y.columns[0].split('_')[-1],
                                  title_text='Backwards Linear Regression Resultant Feature Space: OLS_predicted_' +
                                             Y.columns[0] + ' vs Actual ' +
                                             Y.columns[0] + ': MSE = ' + str(MSE))
            else:
                fig = None

        return y_hat_model, x_test, fig,dropped_feats


    def get_collinearity(self,X,method):
        if method =='VIF':
            # VIF dataframe
            vif_data = pd.DataFrame()
            vif_data["Features"] = X.columns

            # calculating VIF for each feature
            vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                               for i in range(len(X.columns))]

            ExploratoryDataAnalysis().to_pretty_table(dat=vif_data,
                                                      title = 'Variance Inflation Factor (VIF) for Collinearity Analysis',
                                                      head=None)
            return vif_data





    def compare_dimens_reduct_methods(self, model_a, model_b, mod_a_distinct_method, mod_b_distinct_method, mod_a_res, mod_b_res,
                           show_best,target_name):
        df = pd.DataFrame()
        df['Model Predictor Metrics:'] = ['AIC (minimize)', 'BIC (minimize)', 'Adj. R^2 (maximize)']

        mod_a_y_hat_name = mod_a_res.columns[np.where(np.array([i.find('OLS') for i in mod_a_res.columns]) == 0)]
        target_name_pred = mod_a_res[mod_a_y_hat_name].columns[0]
        y_hat = np.array(mod_a_res[mod_a_y_hat_name])
        Y = np.array(mod_a_res[target_name])
        MSE_A = mean_squared_error(Y, y_hat.reshape(len(y_hat), 1))

        mod_b_y_hat_name = mod_b_res.columns[np.where(np.array([i.find('OLS') for i in mod_b_res.columns]) == 0)]
        y_hat = np.array(mod_b_res[mod_b_y_hat_name])
        Y = np.array(mod_b_res[target_name])
        MSE_B = mean_squared_error(Y, y_hat.reshape(len(y_hat), 1))

        df[mod_a_distinct_method] = [model_a.aic, model_a.bic, model_a.rsquared_adj]
        df[mod_b_distinct_method] = [model_b.aic, model_b.bic, model_b.rsquared_adj]

        ExploratoryDataAnalysis.to_pretty_table(self,dat=df,
                             title='Dimensionality Reduction Metric Comparison',
                             head=None)

        if show_best:
            best_wrt_minimize_predictors = df.iloc[0:-1, 1:].idxmin(axis=1).unique()
            r_squared_potential_best = ExploratoryDataAnalysis.slice_by_observation(self,df=df,
                                                                 feature=best_wrt_minimize_predictors,
                                                                 observations=['Adj. R^2 (maximize)'],
                                                                 obs_by_feature=['Model Predictor Metrics:']).iloc[0, 0]
            other_wrt_minimize_predictors = df.iloc[0:-1, 1:].idxmax(axis=1).unique()
            r_squared_expected_worst = ExploratoryDataAnalysis.slice_by_observation(self,df=df,
                                                                 feature=other_wrt_minimize_predictors,
                                                                 observations=['Adj. R^2 (maximize)'],
                                                                 obs_by_feature=['Model Predictor Metrics:']).iloc[0, 0]

            if len(best_wrt_minimize_predictors) == 1 and len(other_wrt_minimize_predictors) == 1 and \
                    r_squared_potential_best >= r_squared_expected_worst:
                # if True:

                rec_method_name = best_wrt_minimize_predictors
                if rec_method_name == mod_a_distinct_method:
                    rec_model = model_a
                    rec_res = mod_a_res
                # else:
                #     rec_model = model_b
                #     rec_res = mod_b_res
                # Plot

                fig = go.Figure()
                predictions = rec_model.get_prediction()
                df_predictions = predictions.summary_frame()
                rec_res['n_observations'] = np.arange(1, len(rec_res) + 1).reshape(len(rec_res), 1)

                fig.add_trace(go.Scatter(name="CI", x=rec_res.n_observations,
                                         y=df_predictions.obs_ci_lower.iloc[:],
                                         line=dict(color='rgba(0, 0, 255, 0.05)'),
                                         fillcolor="#eaecee",
                                         connectgaps=False,
                                         ))
                fig.add_trace(go.Scatter(name=None, x=rec_res.n_observations,
                                         y=df_predictions.obs_ci_upper.iloc[:],
                                         line=dict(color='rgba(0, 0, 255, 0.05)'),
                                         fill="tonexty",
                                         fillcolor="#EAEDF8",
                                         connectgaps=False, showlegend=False
                                         ))

                fig.add_trace(go.Scatter(name=target_name_pred, x=rec_res.n_observations,
                                         y=rec_res[target_name_pred],
                                         line=dict(color='blue')))
                fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
                fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=False)
                fig.update_yaxes(tickfont_family="Arial Black")
                fig.update_xaxes(tickfont_family="Arial Black")
                fig.update_layout(template="plotly_dark", font=dict(size=18))
                fig.update_layout(showlegend=True,
                                  yaxis_title=target_name_pred.split('_')[-1], xaxis_title='# of Samples',
                                  title_text= rec_method_name[0] + 'Regression Results with Confidence Interval (CI) ')
            else:
                rec_method_name = None
                rec_model = None
                fig = None
                print(f'WARNING: OLS predictors do not unanimously recommend a model, use inspection. ')

        return df, fig,df_predictions


class ClassificationAnalysis:
    def __init__(self, info=None):
        pass

    @staticmethod
    def to_one_vs_all(model,x_train, y_train,fit):
        model_ova = OneVsRestClassifier(model)
        if fit:
            model_ova = model_ova.fit(x_train, y_train)
        return model_ova

    @staticmethod
    def to_one_vs_one(model,x_train, y_train,fit):
        model_ovo = OneVsOneClassifier(model)
        if fit:
            model_ovo = model_ovo.fit(x_train, y_train)
        return model_ovo

    @staticmethod
    def get_decision_tree_classifier(x_train, y_train,fit):
        dtf = DecisionTreeClassifier(random_state=123)
        if fit:
            dtf = dtf.fit(x_train, y_train)
        return dtf

    @staticmethod
    def get_random_forest_classifier(x_train, y_train,fit):
        clf = RandomForestRegressor(  random_state=1,
                                      max_depth=5,
                                      n_estimators=100)
        if fit:
            clf = clf.fit(x_train, y_train)
        return clf

    @staticmethod
    def get_logistic_regression_classifier(x_train, y_train,fit):
        clf = LogisticRegression(max_iter=2500,random_state=123)
        if fit:
            clf = clf.fit(x_train, y_train)
        return clf

    @staticmethod
    def show_classification_models( models, methods, results, show):

        fig = None
        fig2 = None
        if show is None:
            show = False

        df = pd.DataFrame()
        df['Classifier Metrics:'] = ['Accuracy (maximize)', 'Precision (maximize)', 'Recall (maximize)',
                                     'F-Score (maximize)']

        traces = []
        c = 0
        fig = go.Figure()
        figure = []
        for mod, meth, res in zip(models, methods, results):
            c += 1
            pred_name = res.columns[np.where(np.array([i.find('Predicted') for i in res.columns]) == 0)]
            target_pred = res[pred_name].columns[0]
            y_hat = np.array(res[target_pred])
            Y = np.array(res[target_pred.split(' ')[-1]])
            precision, recall, f_score, _ = precision_recall_fscore_support(Y, y_hat)
            df[meth] = [accuracy_score(Y, y_hat), precision, recall, f_score]
            cnf_matrix = confusion_matrix(Y, y_hat)

            if show:
                # Predict probabilities of positive class for test set
                probas = mod.predict_proba(res.iloc[:, :-3])[:, 1]

                # Compute ROC curve and AUC
                fpr, tpr, thresholds = roc_curve(Y, probas)
                roc_auc = auc(fpr, tpr)

                # Compute confusion matrix and classification report
                # y_hat = mod.predict(res.iloc[:,:-3])
                cm = confusion_matrix(Y, y_hat)
                cr = classification_report(Y, y_hat)

                # Plot ROC curve
                traces.append(go.Scatter(x=fpr, y=tpr, mode='lines', name=meth + ' (AUC = %0.2f)' % roc_auc))

                # Plot confusion matrix
                fig2 = go.Figure()
                fig2.add_trace((go.Heatmap(colorbar=dict(title='Target: ' + target_pred.split(' ')[-1]), z=cm[::-1],
                                           x=['<b>Predicted Negative<b>', '<b>Predicted Positive<b>'], name=meth,
                                           y=['<b>Actual Positive<b>', '<b>Actual Negative <b>'],
                                           colorscale='viridis',text=cm[::-1],
                                            texttemplate="%{text}",
                                            textfont={"size":20})))


                fig2.update_layout(title_text='<b>Confusion matrix: Classification Model = ' + meth + '<b>',
                                   xaxis_title='<b>Predicted label<b>',
                                   yaxis_title='<b>True label<b>')
                fig2.update_yaxes(tickfont_family="Arial Black")
                fig2.update_xaxes(tickfont_family="Arial Black")
                # fig.update_yaxes(tickfont_family="Arial Black")
                # fig.update_xaxes(tickfont_family="Arial Black")
                fig2.update_layout(template="plotly_dark",font=dict(size=18))
                # fig2.show()

                # # Plot the perfomance
                # f = px.bar(df, x=meth, y='Classifier Metrics:')
                # f.update_yaxes(tickfont_family="Arial Black")
                # f.update_xaxes(tickfont_family="Arial Black")
                # f.show()

        if show:
            fig = go.Figure(data=traces)
            fig.update_layout(xaxis_title='<b>False Positive Rate<b>', yaxis_title='<b>True Positive Rate<b>',
                              title='<b>Receiver Operating Characteristic (ROC) curve<b>')
            fig.update_yaxes(tickfont_family="Arial Black")
            fig.update_xaxes(tickfont_family="Arial Black")
            fig.update_layout(template="plotly_dark",font=dict(size=18))

        ExploratoryDataAnalysis().to_pretty_table(dat=df,
                                                title='Classifier Model Predictor Comparison',
                                                head=None)

        best_wrt_maximize_predictors = df.iloc[0:-1, 1:].idxmax(axis=1)[0]
        ind = int(np.where(np.array([i.find(best_wrt_maximize_predictors) for i in df.columns]) == 0)[0])
        rec_model = models[ind - 1]
        rec_res = results[ind - 1]

        return df, rec_model, rec_res,fig,fig2

    @staticmethod
    def predict_fitted_model(fitted_model, x_test, y_test, compute_prediction,
                             dropped_feats=None, show=False, title=None, dim_red_method=None):
        # Drop the features
        if dropped_feats is not None:
            x_test = x_test.drop(columns=dropped_feats)

        # Compute the fit or use the fitted model given in input
        model = fitted_model
        fig = None

        if compute_prediction:
            # Compute a prediction on the test data set
            # Using Package
            Y = y_test
            y_hat = model.predict(x_test)
            # MSE = mean_squared_error(np.array(Y), np.array(y_hat.reshape(len(y_hat), 1)))
            x_test[Y.columns[0]] = Y
            x_test['Observations'] = np.arange(1, len(x_test) + 1).reshape(len(x_test), 1)
            x_test['Predicted ' + Y.columns[0]] = np.array(y_hat).reshape(len(y_hat), 1)

            # print(f'The mean squared error of the model against the test data is {MSE:.3f}')
            if show:
                # Plot
                fig = px.line(x_test, x_test['Observations'], y=[Y.columns[0], 'Predicted ' + Y.columns[0]])
                fig.update_layout(plot_bgcolor='white')
                fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
                fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=False)
                if title is None:
                    title_str = dim_red_method + ' Resultant Feature Space:  Predicted ' \
                                + Y.columns[0] + ' vs Actual ' + \
                                Y.columns[0]
                else:
                    title_str = title
                    fig.update_layout(showlegend=True,
                                      title_text=title_str,
                                      yaxis_title=Y.columns[0])

        return x_test, fig

    @staticmethod
    def prune_decision_tree(x_train, y_train, x_test, y_test, prune_type, show=False):

        if prune_type == 'pre_prune':
            model = make_pipeline(
                DecisionTreeClassifier()
            )
            # param_grid = [{'max_depth': [1,2,3,4,5,6,7,8,9,10],
            #                'min_samples_split': [1,2,3,4,5],
            #                'min_samples_leaf': [1,2,3,4,5],
            #                'max_features': [2,4,6,8,16,24,36],
            #                'splitter': ['best', 'random'],
            #                'criterion': ['gini', 'entropy', 'log_loss']}]

            param_grid = [{'max_depth': [1, 2],
                           'min_samples_split': [1, 2],
                           'min_samples_leaf': [1, 2, ],
                           'max_features': [2, 4, ],
                           'splitter': ['best', 'random'],
                           'criterion': ['gini', 'entropy', 'log_loss']}]

            dtc = DecisionTreeClassifier(random_state=123)
            # Search over the defined grid
            grid = GridSearchCV(estimator=dtc, param_grid=param_grid, verbose=True)
            model = grid.fit(x_train, y_train)
            dtc_pre_pruned = model.best_estimator_

            if show:
                # Visualizing Decision Tree
                plt.figure(figsize=(16, 8))
                tree.plot_tree(dtc_pre_pruned, rounded=True, filled=True)
                plt.title("Pre-Pruned Tree")
                plt.show()

            return grid.best_params_, dtc_pre_pruned
        elif prune_type == 'post_prune':
            dtc = DecisionTreeClassifier(random_state=123)
            model = dtc.fit(x_train, y_train)
            path = dtc.cost_complexity_pruning_path(x_train, y_train)  # print(path)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            # ccp alpha wil give list of values
            # print("Impurities in Decision Tree :", impurities)

            # Finding an optimal value of alpha
            clfs = []
            for ccp_alpha in ccp_alphas:
                dtc = DecisionTreeClassifier(random_state=123, ccp_alpha=ccp_alpha)
                dtc.fit(x_train, y_train)
                clfs.append(dtc)
            # As we already know that there is a strong relation between, alpha and the depth of the tree.
            # We can find the relation using this plot.
            # print(f"Last node in Decision tree is {clfs[-1].tree_.node_count}
            # and ccp_alpha for last node is {ccp_alphas[-1]}")
            tree_depths = [clf.tree_.max_depth for clf in clfs]
            plt.figure(figsize=(10, 6))
            plt.plot(ccp_alphas[:-1], tree_depths[:-1])
            plt.xlabel("effective alpha")
            plt.ylabel("total depth")
            plt.title("Post-Pruned Tree")

            train_scores = [clf.score(x_train, y_train) for clf in clfs]
            test_scores = [clf.score(x_test, y_test) for clf in clfs]
            fig, ax = plt.subplots()
            ax.set_xlabel("alpha")
            ax.set_ylabel("accuracy")
            ax.set_title("Post-Pruned Tree Accuracy vs alpha for training and testing sets")
            ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
            ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
            ax.legend()

            acc_scores = [accuracy_score(y_test, clf.predict(x_test)) for clf in clfs]
            max_acc_ind = np.argmax(acc_scores)
            dtc_post_pruned_best_alpha = ccp_alphas[max_acc_ind]
            # tree_depths = [dtf.tree_.max_depth for dtf in clfs]
            plt.figure(figsize=(10, 6))
            plt.grid()
            plt.plot(ccp_alphas[:-1], acc_scores[:-1])
            plt.xlabel("effective alpha")
            plt.ylabel("Accuracy scores")

            # The post Pruned tree
            dtc_post_pruned = DecisionTreeClassifier(random_state=123, ccp_alpha=dtc_post_pruned_best_alpha)
            dtc_post_pruned.fit(x_train, y_train)
            plt.figure(figsize=(12, 8))
            tree.plot_tree(dtc_post_pruned, rounded=True, filled=True)
            if show:
                plt.show()

            return dtc_post_pruned_best_alpha, dtc_post_pruned

    @staticmethod
    def grid_search_2D( model_type, x_train, y_train):

        fig = go.Figure()

        if model_type == 'Polynomial regression':
            # Pipeline for polynomial regression
            model = make_pipeline(
                PolynomialFeatures(include_bias=False),
                LinearRegression()
            )
        elif model_type == 'Decision Tree Classifier':
            model = make_pipeline(
                DecisionTreeClassifier()
            )
            # param_grid = [{'max_depth': [1,2,3,4,5,6,7,8,9,10],
            #                'min_samples_split': [1,2,3,4,5],
            #                'min_samples_leaf': [1,2,3,4,5],
            #                'max_features': [2,4,6,8,16,24,36],
            #                'splitter': ['best', 'random'],
            #                'criterion': ['gini', 'entropy', 'log_loss']}]

            param_grid = [{'max_depth': [1, 2],
                           'min_samples_split': [1, 2],
                           'min_samples_leaf': [1, 2, ],
                           'max_features': [2, 4, ],
                           'splitter': ['best', 'random'],
                           'criterion': ['gini', 'entropy', 'log_loss']}]

            dtc = DecisionTreeClassifier(random_state=123)
            # Search over the defined grid
            grid = GridSearchCV(estimator=dtc, param_grid=param_grid, verbose=True)
            # Use the DecisionTreeClassifier to fit the best model to the train data
            model = grid.fit(x_train.values, y_train.values)
            model = model.best_estimator_
            fig = None

        if model_type == 'Polynomial regression':
            for i in np.arange(1, 16):
                # define the parameters to search over
                param_grid = {'polynomialfeatures__degree': [i]}

                # Search over the degree of polynomial features given in param_grid
                grid = GridSearchCV(model, param_grid)
                # Use the LinearRegression to fit each element of the grid
                grid.fit(np.array(x_train).reshape(len(x_train), 1), y_train)
                y_hat = grid.predict(np.array(x_train).reshape(len(x_train), 1))
                RMSE = np.sqrt(np.square(np.subtract(y_train, y_hat)).mean())
                fig.add_trace(go.Scatter(x=np.array([i]), y=np.array(RMSE), marker={'size': 15}, name="n= " + str(i)))
            model = grid.fit(np.array(x_train).reshape(len(x_train), 1), y_train)

            fig.update_layout(plot_bgcolor='white')
            fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
            fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=False)
            fig.update_layout(showlegend=True,
                              yaxis_title='RMSE', xaxis_title='n order polynomial model',
                              title_text='Polynomial Regression grid search for M=1 features linearly dependant features ')

        # Show the best prediction
        return fig, grid.best_params_, model


class ClusteringAnalysis:
    def __init__(self, info=None):
        pass


class AssociationMining:
    def __init__(self, info=None):
        self.info = info

    @staticmethod
    def get_associations(df,show,method ='apriori'):
        frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
        if show:
            # fig = go.Figure(data=[go.Scatter(x=rules['support'], y=rules['lift'],
            #                                  mode='markers', text=rules['antecedents'])])
            # fig.update_layout(title='<b>Association Rules Scatter Plot<b>',
            #                   xaxis_title='<b>Support<b>',
            #                   yaxis_title='<b>Lift<b>')
            # fig.update_yaxes(tickfont_family="Arial Black")
            # fig.update_xaxes(tickfont_family="Arial Black")
            # fig.update_layout(template="plotly_dark",font=dict(size=18))

            def change_to_list(x):
                return list(x)

            rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
            rules['antecedents'] = rules['antecedents'].apply(change_to_list)
            rules['consequents'] = rules['consequents'].apply(change_to_list)
            eda = ExploratoryDataAnalysis()
            eda.to_pretty_table(rules,title='Apriori Association Analysis',head=None)

        return rules

