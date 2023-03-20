import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gmean
from scipy.stats import hmean
from scipy.spatial import distance
from sklearn import metrics
from sklearn.datasets import make_regression,make_classification,make_blobs
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from prettytable import PrettyTable
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor

pd.set_option("display.precision", 2)





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

    def get_aggregate(self,df):
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

        return df_processed

    def show_aggregate(self,df,df_agg,plot_type,title):
        if plot_type == 'plotly':

            # Create the figure
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                specs=[[{"type": "table"}],
                       [{"type": "xy"}]]
            )

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

            df_numeric= self.get_numerical_features(df)
            for i,feat in enumerate(df_numeric.columns):
                fig.add_trace(go.Violin(y=df_numeric[feat],
                                        box_visible=True,
                                        meanline_visible=True,
                                        name=feat),
                                        row=2, col=1)
            fig.update_yaxes(type="log", row=2, col=1)



            fig.update_layout(
                showlegend=True,
                title_text=title,
            )
            return fig




    def get_percent_of_missing_observations(self,df,clean, show,clean_method):
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
                df = df.drop(columns = dirty_feats)
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


    def normalize(self,df):
        df_init = self.get_numerical_features(df)
        feat_names = list(df_init.columns)
        df_norm_columns = [f'{df_init.columns[i]} (normalized)' for i in range(0, df_init.shape[1])]
        x_norm_mat = np.zeros((df_init.shape[0],df_init.shape[1]))
        for i,fn in enumerate(feat_names):
            x = df_init[fn]
            x_norm_mat[:,i] = (x - np.min(x))/(np.max(x) - np.min(x))
        df_norm = pd.DataFrame(x_norm_mat,columns=df_norm_columns)
        return df_norm

    def standardize(self,df):
        df_init = self.get_numerical_features(df)
        feat_names = list(df_init.columns)
        df_stand_columns = [f'{df_init.columns[i]} (standardized)' for i in range(0, df_init.shape[1])]
        x_stand_mat = np.zeros((df_init.shape[0],df_init.shape[1]))
        for i,fn in enumerate(feat_names):
            x = df_init[fn]
            x_stand_mat[:,i] = (x - np.mean(x))/np.std(x)
        df_stand = pd.DataFrame(x_stand_mat,columns=df_stand_columns)
        return df_stand


    def iq_transform(self,df):
        df_init = self.get_numerical_features(df)
        feat_names = list(df_init.columns)
        df_iqt_columns = [f'{df_init.columns[i]} (iqt)' for i in range(0, df_init.shape[1])]
        x_iqt_mat = np.zeros((df_init.shape[0], df_init.shape[1]))
        for i, fn in enumerate(feat_names):
            x = df_init[fn]
            x_iqt_mat[:,i] = (x - x.quantile(.5))/(x.quantile(.75) - x.quantile(.25))
        df_iqt = pd.DataFrame(x_iqt_mat, columns=df_iqt_columns)
        return df_iqt

    def minkowski_distance(self,x, y, L):
        m_dist = np.zeros((len(x),1))
        for i in range(0,len(x)):
            sum_abs_raised = sum(abs(np.array([x[i],y[i]]))**L)
            m_dist[i] = sum_abs_raised**(1/L)
        return m_dist.reshape(len(m_dist),1)


    def plot_all_features(self,df,x_feature,df_name,on_subplots,plot_on_this):
        if x_feature is not None:
            df['Date'] = x_feature
        if len(df.shape) >1:
            n_plots = df.shape[1]
            n_col = int(n_plots/2)
            if n_plots % 2 ==0:
                n_row = (n_plots - n_col)
            else:
                n_row = (n_plots - n_col)+1
            left =n_row*n_col-n_plots
            if left>0 and left>n_col:
                n_row = n_row-1

        else:
            n_plots = 1
            n_row = 1
            n_col = 1

        if plot_on_this is not None:
            df.plot(x='Date',kind='line', subplots=on_subplots, grid=True, title=df_name,
                    layout=(n_row, n_col), sharex=False, sharey=False, legend=True,ax=plot_on_this)
        else:
            df.plot(x='Date', kind='line', subplots=on_subplots, grid=True, title=df_name,
                    sharex=False, sharey=False, legend=True)

            # df.plot(subplots=on_subplots)
            # plt.tight_layout()
            plot_on_this = plt.gcf().axes

        for i,ax in enumerate(plt.gcf().axes):
            if i == len(plt.gcf().axes)-1:
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

    def get_pca(self,df):
        pca = PCA(n_components=len(df.columns),svd_solver='full')
        pca.fit(df)
        X_PCA = pca.transform(df)
        return X_PCA

    def plot_features_unified_xaxis(self,df,x_axis_feature,y_axis_feature,observation_ID,plot_type,title):
        plt.figure(figsize=(12, 8))
        legFlag = True
        for obs in observation_ID:
            for i,yy in enumerate(y_axis_feature):
                # x = df[df['Symbol'] == obs][x_axis_feature]
                # y = df[df['Symbol'] == obs][y_axis_feature]

                # if y.isnull().values.any():
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
                            x.iloc[int(1*(n_samples/6))],
                            x.iloc[int(2*(n_samples/6))],
                            x.iloc[int(3*(n_samples/6))],
                            x.iloc[int(4*(n_samples/6))],
                            x.iloc[int(5*(n_samples/6))],
                            x.iloc[int(6 * (n_samples / 6))-1]
                             ]))

    def slice_by_observation(self,df,feature,observations,obs_by_feature):
        cmmd_str = ''
        temp = np.zeros((df.shape[0],1))
        df_temp = pd.DataFrame(temp)
        if observations is not None:
            obs = observations[0]
            obs_by_feat = obs_by_feature[0]
        for i,feat in enumerate(feature):
            cmmd_str = ' '"'" + feat + "'"' '
            if observations is None:
                exec('df_temp[' + cmmd_str + '] =  df[' + cmmd_str + ']')
            else:
                for obs in observations:
                     # df['Symbol'].loc[df['Symbol'] == 'A']
                    exec('df_temp[' + cmmd_str + '] =  df[' + cmmd_str + '].loc[df['"'"+ obs_by_feat +"'" '] == '"'"+ obs +"'" ']')
        df_send = df_temp.iloc[:,1:]
        df_send_trunc = df_send.dropna()
        return df_send_trunc

    def compute_skewness(self,data, order):
        N = len(data)
        m_i = (1/N)*np.sum((data - np.mean(data))**order)
        return m_i

    def to_pretty_table(self,dat,title,head):
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

    def plot_data_with_principal_axes(self,df,ds_title):
        data = df.values
        mu = data.mean(axis=0)
        data = data - mu
        cov = np.cov(df, rowvar=False)
        eigenvalues,eigenvectors = np.linalg.eig(cov)
        projected_data = np.dot(data, eigenvectors)
        sigma = projected_data.std(axis=0).mean()


        fig, ax = plt.subplots()
        colors = ['red','blue']
        plt.scatter(data[:,0], data[:,1])
        norm_eigen = normalize(eigenvalues.reshape(1,len(eigenvalues))) + (1-np.max(normalize(eigenvalues.reshape(1,len(eigenvalues)))))
        for i,axis in enumerate(eigenvectors):
            end = 0 + (sigma * np.multiply(axis, *norm_eigen[:, i]))
            end[0] = end[0] * -1

            start = 0, 0
            plt.scatter(start[0],start[1],color=colors[i],label = 'Feature: '+df.columns[i]+ ' - Eigenvalue = ' + str(round(eigenvalues[i],2)),marker = '<',s = 1)

            ax.annotate(
                '', xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(facecolor=colors[i], width=2.0))
        ax.set_aspect('equal')

        plt.xlabel(df.columns[0] + 'rotated into Eigenspace')
        plt.ylabel(df.columns[1] + 'rotated into Eigenspace')
        plt.title(ds_title)
        plt.legend()

    def time_difference(self,t,order):
            diff = list()
            for i in range(order, len(t)):
                value = t[i] - t[i - order]
            diff.append(value)
            return np.array(diff).reshape(len(diff),1)

    def show_random_forest_analysis(self,X,y,rank_list,plot_type,title):

        model = RandomForestRegressor(random_state=1,
                                      max_depth=5,
                                      n_estimators=100)

        if self.get_num_categorical_features(y) ==1:
            y['rank'] = None
            for index, row in y.iterrows():
                y['rank'][index] = rank_list.index(row['target_buy_sell_performance'])


        model.fit(X, y['rank'])


        features = X.columns
        importance = model.feature_importances_
        indices = np.argsort(importance)[-len(importance):]  # top 20

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
                                  tickvals=np.arange(0,len(indices)),
                                  ticktext=[features[i] for i in indices]),
                              showlegend = True,
                              title_text = title,
                              )
            # ticktext = [range(len(indices)), [features[i] for i in indices]]
            # for idx in range(len(fig.data)):
            #     fig.data[idx].y = ticktext[idx]




            return fig









