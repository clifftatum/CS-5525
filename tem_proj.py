import pandas as pd
from matplotlib import pyplot as plt
from src.EDA import EDA
import seaborn as sb
import plotly.express as px
import numpy as np
np.random.seed(5525)
import sklearn

sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum EDA Deflation Market Nov 2022 - Jan 2023 - CS-5525')
    url = 'https://raw.githubusercontent.com/clifftatum/CS-5525-Term-Project/main/DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv'
    df = pd.read_csv(url)

    # Change the date feature to a datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the date by oldest to newest
    df.sort_values(by='Date', inplace=True,ascending = True)

    # Prune the useless Features
    df.drop(columns='Unnamed: 0',inplace=True)
    num_missing, percent_removed, df_clean = eda.get_percent_of_missing_observations(df,clean=True,
                                                                                        show=True,
                                                                                        clean_method='prune')
    df = df_clean

    # Aggregation
    df_agg = eda.get_aggregate(df)
    fig = eda.show_aggregate(df,df_agg,plot_type = 'plotly',title = ' Aggregate analysis:  Deflation Market Nov 2022 - Jan 2023')



    # Dimensionality reduction / feature selection:
    # Random forest

    # One hot encode
    # label_binarizer = sklearn.preprocessing.LabelBinarizer()
    # X = df.drop(columns='Date')
    # X = label_binarizer.fit_transform(X)

    fig = eda.show_random_forest_analysis(X=eda.get_numerical_features(df),
                                          y=eda.slice_by_observation(df, ['target_buy_sell_performance'],
                                                                     observations=None, obs_by_feature=None),
                                          rank_list=['great_buy', 'great_sell', 'good_buy', 'good_sell', 'Hold',
                                                     'poor_sell', 'poor_buy', 'horrible_sell', 'horrible_buy'],
                                          plot_type='plotly',
                                          title = ' Random Forest Analysis:  Deflation Market Nov 2022 - Jan 2023')


    fig.show()


    # model.fit(X,y)
    # features = df.columns
    # importance = model.feature_importances_
    # indices = np.argsort(importance)[-20:]
    # plt.barh(range(len(indices)), importance[indices], color='b',align='center')
    # plt.yticks(range(len(indices)), [features[i] for i in indices])
    # plt.xlabel('Relative Importance')
    # plt.tight_layout()
    # plt.show()














    # eda.plot_features_unified_xaxis(df_agg,x_axis_feature=['Date'])

    pass








    # fig = px.scatter(df, x="SlowPctK", y="PctB", color="target_buy_sell_performance", marginal_y="violin",
    #                  marginal_x="box", trendline="ols", template="simple_white")
    # fig.show()

    # df.drop(['price_pct_change'],axis1,inplace=True)


