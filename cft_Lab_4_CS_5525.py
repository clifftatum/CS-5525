import pandas as pd
from matplotlib import pyplot as plt
from src.EDA import EDA
import seaborn as sb
import numpy as np


sb.set(color_codes=True)
pd.set_option("display.precision", 3)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lab 4 - CS-5525')
    url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Carseats.csv'
    df = pd.read_csv(url)

    # Problem 1)A
    fig1 = eda.show_hbar(df,
                     x_feat='Sales',
                     y_feat='ShelveLoc',
                     by_leg_category='US')

    df_agg = eda.get_aggregate(df)

    df_yes_us = eda.slice_by_observation(df,feature=['Sales','ShelveLoc','US'],
                                      observations=['Yes'],
                                      obs_by_feature=['US'])

    df_no_us = eda.slice_by_observation(df,feature=['Sales','ShelveLoc','US'],
                                      observations=['No'],
                                      obs_by_feature=['US'])

    df_yes_us_sums_by_shelvloc = df_yes_us.groupby(['ShelveLoc']).sum()
    df_yes_us_sums_by_shelvloc.insert(0, 'ShelveLoc', df_yes_us_sums_by_shelvloc.T.columns)

    df_no_us_sums_by_shelvloc = df_no_us.groupby(['ShelveLoc']).sum()
    df_no_us_sums_by_shelvloc.insert(0, 'ShelveLoc', df_no_us_sums_by_shelvloc.T.columns)


    eda.to_pretty_table(dat=df_yes_us_sums_by_shelvloc,
                        title = 'Total Sales within the US by Shelf Location',
                        head = None)
    eda.to_pretty_table(dat=df_no_us_sums_by_shelvloc,
                        title='Total Sales outside the US by Shelf Location',
                        head = None)
    # Problem 1)B
    df_encoded,encoded_ind = eda.one_hot_encode(df)

    # Problem 1)C
    df_encoded.iloc[:,encoded_ind] = eda.standardize(df_encoded.iloc[:,encoded_ind])
    x_train,x_test,y_train,y_test = eda.split_80_20(df_encoded, target="Sales")

    # Problem 2
    x_train = eda.backward_linear_regression(x_train = x_train,
                                         y_train = y_train,
                                         show=True)


    fig1.show()
    pass
