import pandas as pd
from matplotlib import pyplot as plt
from EDA import EDA
import seaborn as sb
import numpy as np


sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

#######################################################################################################################
# This script categorizes and saves a deflation macro-environment dataset to a target feature named:
# 'target_signal_performance'

# The output of this script us used for CS-5525 Term project, file name:
# DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv
#######################################################################################################################


if __name__ == '__main__':
    eda = EDA('Clifford Tatum Term Project - DATASET PREPROCESSING - CS-5525')
    url = 'deflation_market-dataset_2023_cliffordt_CS5525_term_dataset.csv'
    df = pd.read_csv(url)


    # Prune the features not included in our discussion
    df_buy_sells_signals = eda.slice_by_observation(df,
                                      feature=['SSCrossover','price_pct_change'],
                                      observations=None,
                                      obs_by_feature= None)
    gain_range = np.max(df_buy_sells_signals['price_pct_change'])
    loss_range = np.min(df_buy_sells_signals['price_pct_change'])
    num_target_cat = 4
    bin_factor = 1 / 4

    excellent_gain = (gain_range - 0) * (bin_factor * 4)
    significant_gain = (gain_range - 0) * (bin_factor * 3)
    moderate_gain = (gain_range - 0) * (bin_factor * 2)
    minimal_gain = (gain_range - 0) * (bin_factor * 1)
    origin = 0
    minimal_loss = (loss_range - 0) * (bin_factor * 1)
    moderate_loss = (loss_range - 0) * (bin_factor * 2)
    significant_loss = (loss_range - 0) * (bin_factor * 3)
    severe_loss = (loss_range - 0) * (bin_factor * 4)

    def change(x):
        excellent_gain = 337.56
        significant_gain =253.17
        moderate_gain = 168.78
        minimal_gain = 84.39
        origin = 0
        minimal_loss = -13.06
        moderate_loss =-26.13
        significant_loss =-39.16
        severe_loss =-52.26


        if x> origin and x < minimal_gain:
            return 'minimal_gain'
        elif x > minimal_gain and x < moderate_gain:
            return 'moderate_gain'
        elif x > moderate_gain and x < significant_gain:
            return 'significant_gain'
        elif x > significant_gain and x <= excellent_gain:
            return 'excellent_gain'
        elif x < origin and x > minimal_loss:
            return 'minimal_loss'
        elif x < minimal_loss and x > moderate_loss:
            return 'moderate_loss'
        elif x < moderate_loss and x > significant_loss:
            return 'significant_loss'
        elif x < significant_loss and x >= severe_loss:
            return 'severe_loss'
        else:
            return 'Hold'



    df['target_signal_performance'] = df['price_pct_change'].apply(change)



    # df_official['price_pct_change'].fillna(0)
    df.rename(columns={'PriceDate':'Date'},inplace=True)
    df.drop(['DateTime'],axis=1,inplace=True)
    df.drop(['Unnamed: 0'], axis=1,inplace=True)

    df['target_signal_performance'].value_counts().plot()


    df_exel_g = eda.slice_by_observation(df,
                                      feature=['price_pct_change','target_signal_performance'],
                                      observations=['excellent_gain'],
                                      obs_by_feature= ['target_signal_performance'])


    df_sig_g = eda.slice_by_observation(df,
                                      feature=['price_pct_change','target_signal_performance'],
                                      observations=['significant_gain'],
                                      obs_by_feature= ['target_signal_performance'])

    df_mod_g = eda.slice_by_observation(df,
                                      feature=['price_pct_change','target_signal_performance'],
                                      observations=['moderate_gain'],
                                      obs_by_feature= ['target_signal_performance'])

    df_min_g = eda.slice_by_observation(df,
                                      feature=['price_pct_change','target_signal_performance'],
                                      observations=['minimal_gain'],
                                      obs_by_feature= ['target_signal_performance'])


    df_severe_l = eda.slice_by_observation(df,
                                      feature=['price_pct_change','target_signal_performance'],
                                      observations=['severe_loss'],
                                      obs_by_feature= ['target_signal_performance'])


    df_sig_l = eda.slice_by_observation(df,
                                      feature=['price_pct_change','target_signal_performance'],
                                      observations=['significant_loss'],
                                      obs_by_feature= ['target_signal_performance'])

    df_mod_l = eda.slice_by_observation(df,
                                      feature=['price_pct_change','target_signal_performance'],
                                      observations=['moderate_loss'],
                                      obs_by_feature= ['target_signal_performance'])

    df_min_l = eda.slice_by_observation(df,
                                      feature=['price_pct_change','target_signal_performance'],
                                      observations=['minimal_loss'],
                                      obs_by_feature= ['target_signal_performance'])



    plt.ylim([0,100])
    plt.show()


    df.to_csv('DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv')

