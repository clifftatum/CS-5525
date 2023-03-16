import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class data_preprocessing:
    def __init__(self):
        pass
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

    ####### plot_all_features is a generalized method that
    ####### replaces show_original, show_normalized,show_standardized, show_Iqr
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
                    layout=(n_row, n_col), sharex=False, sharey=False, legend=True)
            plot_on_this = plt.gcf().axes

        for i,ax in enumerate(plt.gcf().axes):
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

    # helper method for norm,stand, and IQT
    def get_numerical_features(self,df,sb_name=None):

        feat_names = list(df.columns)
        num_feat_names = []
        for fn in feat_names:
            feat = df[fn]
            d_type = str(feat.dtypes)
            if d_type == "int64" or d_type == 'float64':
                num_feat_names.append(fn)
        return df[num_feat_names]
