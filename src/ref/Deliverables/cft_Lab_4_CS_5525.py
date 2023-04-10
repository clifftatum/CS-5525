import pandas as pd
from matplotlib import pyplot as plt
from src.EDA import EDA
import seaborn as sb
import numpy as np
np.set_printoptions(precision=3)



sb.set(color_codes=True)
pd.set_option("display.precision", 3)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lab 4 - CS-5525')
    url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Carseats.csv'
    df = pd.read_csv(url)
    show_plot = True

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
    df_encoded,encoded_ind = eda.one_hot_encode(df.copy(deep=True))
    print(df_encoded.iloc[:5,encoded_ind].head())

    # Problem 1)C
    df_encoded_sub_standardized = df_encoded.copy(deep=True)
    temp = df_encoded_sub_standardized.drop(columns=df_encoded_sub_standardized.columns[encoded_ind])
    df_encoded_sub_standardized = eda.standardize(df=temp,
                                                  compute_method='manual')
    df_encoded_sub_standardized[df_encoded.columns[encoded_ind]] = df_encoded[df_encoded.columns[encoded_ind]]
    x_train,x_test,y_train,y_test = eda.split_80_20(df=df_encoded_sub_standardized.copy(deep=True),
                                                    target="Sales")
    print(f'Train dataset')
    print(x_train.head())
    print(f'Test dataset')
    print(x_test.head())
    # Problem 2
    OLS_model_BLR,results_BLR,fig2 = eda.backward_linear_regression( x_train = x_train.copy(deep=True),
                                                             y_train = y_train.copy(deep=True),
                                                             x_test = x_test.copy(deep=True),
                                                             y_test = y_test.copy(deep=True),
                                                             compute_prediction=True,
                                                             compute_method='package',
                                                             show=True)


    # Problem 3
    df_enc_no_target = df_encoded.copy(deep=True)
    df_enc_no_target.drop(columns = 'Sales',inplace=True)

    # df_standardized_no_target = eda.standardize(df=df_enc_no_target,
    #                                             compute_method='manual')
    df_standardized_no_target = df_encoded_sub_standardized.drop(columns=['Sales'])

    n_req_features_for90_perc_exp_var,fig3 = eda.get_pca(df=df_standardized_no_target,
                                                         show_cum_exp_var=True,
                                                         required_exp_variance=0.9)

    # Problem 4 A)
    fig4,drop_these = eda.random_forest_analysis(X=df_standardized_no_target,
                                          y=df_encoded['Sales'],
                                          max_features = n_req_features_for90_perc_exp_var,
                                          rank_list=None,
                                          plot_type='plotly',
                                          title = ' Random Forest Analysis:  Car seats ')

    # Problem 4c through e
    OLS_model_RFA,results_RFA,fig5 = eda.drop_and_show_OLS_prediction(x_train = x_train.copy(deep=True),
                                                                     y_train = y_train.copy(deep=True),
                                                                     x_test = x_test.copy(deep=True),
                                                                     y_test = y_test.copy(deep=True),
                                                                     dropped_feats=drop_these,
                                                                     show=True,
                                                                     title=None,
                                                                     compute_prediction=True,
                                                                     compute_method='package',
                                                                     dim_red_method='Random Forest Analysis')
    # Problem 5 and 6
    df_comp,fig6 = eda.compare_OLS_models(model_a=OLS_model_BLR,
                                     model_b=OLS_model_RFA,
                                     mod_a_distinct_method='Backward Linear Regression',
                                     mod_b_distinct_method='Random Forest Analysis',
                                     mod_a_res=results_BLR,
                                     mod_b_res=results_RFA,
                                     show_best = True)

    # Problem 7
    fig7,best_poly_degree = eda.poly_grid_search_2D(X=df_standardized_no_target['Price'],
                                        indep_feat ='Price' ,
                                        y=df_encoded['Sales'],
                                        dep_targ = 'Sales')


    from sklearn.preprocessing import PolynomialFeatures
    polynomial_features = PolynomialFeatures(degree=best_poly_degree['polynomialfeatures__degree'])

    x_train,x_test,y_train,y_test = eda.split_80_20(df=df_encoded_sub_standardized.copy(deep=True)[['Sales','Price']],
                                                    target="Sales")
    x_train = polynomial_features.fit_transform(x_train)
    OLS_model_poly,results_poly,fig8 = eda.drop_and_show_OLS_prediction(  x_train = x_train,
                                                                     y_train = y_train,
                                                                     x_test = x_test,
                                                                     y_test = y_test,
                                                                     dropped_feats=None,
                                                                     show=True,
                                                                     title='Polynomial regression model - Sales prediction per the Price',
                                                                     compute_prediction=True,
                                                                     compute_method='package',
                                                                     dim_red_method=None)


    # if show_plot:
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()
    fig6.show()
    fig7.show()
    fig8.show()

