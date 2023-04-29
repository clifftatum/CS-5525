import pandas as pd
from src.data_analytics import ExploratoryDataAnalysis, RegressionAnalysis, \
    ClassificationAnalysis, ClusteringAnalysis, AssociationMining
import numpy as np
np.random.seed(5525)

pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = ExploratoryDataAnalysis('Clifford Tatum ExploratoryDataAnalysis Deflation Market Nov 2022 - Jan 2023 - CS-5525')
    ra = RegressionAnalysis()
    am = AssociationMining()
    ca = ClassificationAnalysis()

    url = 'https://raw.githubusercontent.com/clifftatum/CS-5525-Term-Project/main/DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv'
    df_init = pd.read_csv('C:\\Users\\cft5385\\'
                     'Documents\\Learning\\'
                     'CS-5525\\Code\\Term Project\\'
                     'CS-5525-Term-Project\\DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv')

    fig_pack = []
    truncate=False
    show=False
    if truncate:
        df_init = df_init.iloc[1:100, :]
    ####################################################################################################################
    # Phase I: EDA
    ####################################################################################################################

    # Change the date feature to a datetime
    df_init['Date'] = pd.to_datetime(df_init['Date'])

    # Sort the date by oldest to newest
    df_init.sort_values(by='Date', inplace=True, ascending = True)

    # Delete the useless/redundant features(s)
    df_init.drop(columns=['Unnamed: 0', 'target_signal_performance'], inplace=True)
    num_missing, percent_removed, df_init = eda.get_percent_of_missing_observations(df_init.copy(deep=True),
                                                                                    clean=True,
                                                                                    show=True,
                                                                                    clean_method='prune')
    # Make some of the feature's Names more detailed rather than just acronyms
    def change_s_type(x):
        if x=='E':
            return 'ETF'
        elif x=='U':
            return 'Currency'
        elif x=='F':
            return 'FixedIncome'
        elif x=='M':
            return 'Commodity'
        elif x =='S':
            return 'StockEquity'
        elif x =='I':
            return 'Index'
    df_init['StockType'] = df_init['StockType'].apply(change_s_type)
    def change_ssc(x):
        if x=='T':
            return 'Topping'
        elif x=='B':
            return 'Bottoming'
        elif x=='S':
            return 'Sell'
        elif x=='Y':
            return 'Buy'
        elif x =='I':
            return 'Intermediate'
    df_init['SSCrossover'] = df_init['SSCrossover'].apply(change_ssc)
    def change_e(x):
        if x=='R':
            return 'Press'
        elif x=='O':
            return 'Short'
        elif x =='I':
            return 'Intermediate'
    df_init['SSEmbedded'] = df_init['SSEmbedded'].apply(change_e)
    def change_f(x):
        if x=='K':
            return 'Breakout'
        elif x=='X':
            return 'Exit'
        elif x =='I':
            return 'Intermediate'
    df_init['SSFailure'] = df_init['SSFailure'].apply(change_f)



    # Print the target name:
    target_name =df_init.columns[-1]
    print(f'Target: {target_name}')

    # For now, I'm electing to not Aggregate given that it is computational manageable

    # I am electing to Down-sample the Categorical Feature: 'Industry' which has 327 unique categories.
    # Because the chosen encoding method for converting categorical data to numerical data is one-hot encoding,
    # which increases the dimensionality of the data I elect to remove Industry as a feature,
    # but will keep it with my data set

    # Identify the columns which are not features but should remain with the dataset
    not_features = ['Date','Industry','Symbol',target_name]


    # One-hot-encoding converts the qualitative features to quantitative feature
    df_encoded,_,encoded_feats = eda.one_hot_encode(df = df_init.copy(deep=True),
                                      not_features=not_features)
    # print(df_encoded.iloc[:5,encoded_ind].head())

    # Move the target to the end of the meaningful data
    column_to_move = df_encoded.pop(target_name)
    df_encoded.insert(len(df_encoded.columns), target_name, column_to_move)

    # Partition the dataset
    non_feats = df_encoded.columns[0:3]
    feats_to_standardize = df_encoded.columns[3:18]
    feats_encoded = df_encoded.columns[18:-1]

    # Detect and Remove outliers using the Local Outlier Factor (LOF) method
    keep_ind, perc_detected_removed = eda.detect_and_remove_outliers(df = df_encoded.copy(deep=True)[feats_to_standardize],
                                                                     method = '1ClassSVM')
    # Remove the outliers
    df_encoded = df_encoded.iloc[keep_ind.T,:]

    fig0 = eda.show_aggregate(df=df_encoded[feats_to_standardize], df_agg=None,
                              plot_type='plotly',
                              title='<b> Numerical Feature Distribution after Outlier Detection and Removal <b>',
                              split = ['Volume','VolumeTenMA'])
    fig_pack.append(fig0)

    # Standardize (not normalize) the appropriate data
    df_encoded[feats_to_standardize] = eda.standardize(df=df_encoded[feats_to_standardize].copy(deep=True))




    # I should standardize the data, since normalization is not appropriate when handleing numerical features that have
    # significantly different scales or units. Standardization will scale the data to have mean 0 and standard
    # deviation 1, which will allow you to compare and combine the variables on a common scale

    df = df_encoded.iloc[:,len(non_feats):]

    # # Principal Component Analysis (PCA)
    # n_req_features_for90_perc_exp_var_pca, fig1 = eda.get_pca(df=df.copy(deep=True).drop(columns = target_name),
    #                                                           show_cum_exp_var=True,
    #                                                           required_exp_variance=0.90)
    # fig_pack.append(fig1)
    #
    # # Singular Value Decomposition (SVD)
    # n_req_features_for90_perc_exp_var_svd, fig2 = eda.get_svd(df=df.copy(deep=True).drop(columns = target_name),
    #                                                           show_cum_exp_var=True,
    #                                                           required_exp_variance=0.90)
    # fig_pack.append(fig2)
    #
    # # Show the Principle Axes Analysis

    # Both PCA and SVD show that only 7/45 features are required for 90% explained variance,
    # however, 20/45 features provides 98% explained variance, move forward with 20 as the keep threshold



    # Random Forest Analysis (RFA)
    fig3, drop_these_rfa = eda.random_forest_analysis(X=df.copy(deep=True).drop(columns = target_name),
                                                  y=df.copy(deep=True)[target_name],
                                                  max_features = 20,
                                                  plot_type='plotly',
                                                  title = '<b>Random Forest Analysis<b>')
    fig_pack.append(fig3)
    # wait to drop until pre-processing is complete
    # df.drop(columns = drop_these_rfa,inplace=True)



    # Show the Covariance and Pearson Correlation Coefficient Matrices
    fig4 = eda.show_cov(df =df.copy(deep=True).drop(columns = target_name))
    fig_pack.append(fig4)

    fig5 = eda.show_corr(df =df.copy(deep=True).drop(columns = target_name))
    fig_pack.append(fig5)

    ####################################################################################################################
    # Phase II: Regression Analysis
    ####################################################################################################################


    # Get the ANOVA and Chai-Squared analysis for the model
    #
    # I have a categorical response variable (y), therefore the t-test and f-test are not appropriate.
    # These tests assume a continuous response variable, and do not work well with categorical data.
    # The ANOVA test (Analysis of Variance) can be used to test whether there are significant differences between
    # the means of two or more groups. Therefore, I can test whether there are significant differences
    # in the mean values of a continuous predictor variable across different levels of a categorical response variable.

    anove_res = eda.test_null_hypothosis(X=df.copy(deep=True).drop(columns = target_name),
                                         y=df.copy(deep=True)[target_name])


    # Association Analysis - for now use the initial categorical features fod the association analysis
    df_assoc = df_encoded[encoded_feats]
    df_assoc_res = am.get_associations(df=df_assoc, show=True, method='apriori')

    # Collinearity Analysis
    df_VIF = ra.get_collinearity(X=df.copy(deep=True).drop(columns = target_name),method='VIF')
    # The results of Collinearity Analysis make sense, Price specific features, like Open, High, Low, Moving Average are
    # very co-linear.


    # Split the Dataset to train and test (80%-20%) with shuffle=False
    x_train,x_test,y_train,y_test = eda.split_80_20(df=df.copy(deep=True),
                                                    target=target_name)

    # Backward Linear Regression dimensionality reduction
    _,_,_,drop_these_blr = ra.backward_linear_regression( x_train = x_train.copy(deep=True),
                                                                 y_train = y_train.copy(deep=True),
                                                                 x_test = x_test.copy(deep=True),
                                                                 y_test = y_test.copy(deep=True),
                                                                 compute_prediction=True,
                                                                 compute_method='package',
                                                                 show=True,
                                                                 encode_target=True)

    # Confidence Interval Analysis Random Forest Analysis vs Backward Linear Regression
    OLS_model_RFA,results_RFA,_ = ra.drop_and_show_regression_results(x_train=x_train.copy(deep=True),
                                                                      y_train=y_train.copy(deep=True),
                                                                      x_test=x_test.copy(deep=True),
                                                                      y_test=y_test.copy(deep=True),
                                                                      dropped_feats=drop_these_rfa, show=True,
                                                                      title=None, compute_prediction=True,
                                                                      compute_method='package',
                                                                      dim_red_method='Random Forest Analysis')

    OLS_model_BLR,results_BLR,_ = ra.drop_and_show_regression_results(x_train=x_train.copy(deep=True),
                                                                      y_train=y_train.copy(deep=True),
                                                                      x_test=x_test.copy(deep=True),
                                                                      y_test=y_test.copy(deep=True),
                                                                      dropped_feats=drop_these_blr, show=True,
                                                                      title=None, compute_prediction=True,
                                                                      compute_method='package',
                                                                      dim_red_method='Backward Linear Regression')




    df_final_res, fig6,df_predictions = ra.compare_dimens_reduct_methods(model_a=OLS_model_BLR, model_b=OLS_model_RFA,
                                                                         mod_a_distinct_method='Backward Linear Regression',
                                                                         mod_b_distinct_method='Random Forest Analysis',
                                                                         mod_a_res=results_BLR, mod_b_res=results_RFA,
                                                                         show_best=True, target_name=target_name)
    fig_pack.append(fig6)

    ####################################################################################################################
    # Phase III: Classification
    ####################################################################################################################
    # For now select Backward Linear Regression as the method used for dimensionality reduction
    x_test.drop(columns=drop_these_blr,inplace=True)
    x_train.drop(columns=drop_these_blr, inplace=True)


    # Create a 'Full' Decision Tree Classifier - no pruning
    full_dtc = ca.get_decision_tree_classifier(x_train=x_train.copy(deep=True),
                                               y_train=y_train.copy(deep=True),
                                               fit = False)

    # Convert the Full Decision Tree Classifier Model to a One Versus All multi-label classifier
    full_dtc_ova_fitted = ca.to_one_vs_all(full_dtc,x_train=x_train.copy(deep=True),
                                               y_train=y_train.copy(deep=True),
                                               fit = True)

    # Convert the Full Decision Tree Classifier Model to a One Versus One multi-label classifier
    full_dtc_ovo_fitted = ca.to_one_vs_one(full_dtc,x_train=x_train.copy(deep=True),
                                               y_train=y_train.copy(deep=True),
                                               fit = True)

    # Create a Random Forest Classifier
    rfc_fitted = ca.get_random_forest_classifier(x_train=x_train.copy(deep=True),
                                               y_train=y_train.copy(deep=True),
                                               fit = True)

    # Convert the Random Forest Classifier Model to a One Versus One multi-label classifier
    rfc_ovo_fitted = ca.to_one_vs_one(model=rfc_fitted,
                                       x_train=x_train.copy(deep=True),
                                       y_train=y_train.copy(deep=True),
                                       fit=True)

    # Convert the Random Forest Classifier Model to a One Versus All multi-label classifier
    rfc_ova_fitted = ca.to_one_vs_all(model=rfc_fitted,
                                       x_train=x_train.copy(deep=True),
                                       y_train=y_train.copy(deep=True),
                                       fit=True)







    # Make a prediction using the full tree  - One Versus all Classification
    results_dtc_full_tree_ova, _ = ca.predict_fitted_model( fitted_model=full_dtc_ova_fitted,
                                                               x_test=x_test.copy(deep=True),
                                                               y_test=y_test.copy(deep=True),
                                                               compute_prediction=True,
                                                               show=False,
                                                               title=None,
                                                               dim_red_method='Backward Linear Regression')

    # Make a prediction using the full tree  - One Versus One Classification
    results_dtc_full_tree_ovo, _ = ca.predict_fitted_model( fitted_model=full_dtc_ovo_fitted,
                                                               x_test=x_test.copy(deep=True),
                                                               y_test=y_test.copy(deep=True),
                                                               compute_prediction=True,
                                                               show=False,
                                                               title=None,
                                                               dim_red_method='Backward Linear Regression')








    # models = [full_dtc_ova_fitted,full_dtc_ovo_fitted]
    # mod_dist_meths = ['Full Decision Classifier Tree (One vs. All)','Full Decision Classifier Tree (One vs. One)']
    # mod_res = [results_dtc_full_tree_ova,results_dtc_full_tree_ovo]

    # Compare the Classifiers # TODO, the metrics are delivered currently for binary lable classifiers, fix this!
    # df_metrics_full_dtc,rec_model,rec_res = ca.show_classification_models(models=models,
    #                                                               methods=mod_dist_meths,
    #                                                               results=mod_res,
    #                                                               show=False)





    if show:
        for f in fig_pack:
            f.show()

    pass



    # T-test and F-test is lecture 6



    # Split to train, test 80-20






    # fig = px.scatter(df, x="SlowPctK", y="PctB", color="target_buy_sell_performance", marginal_y="violin",
    #                  marginal_x="box", trendline="ols", template="simple_white")
    # fig.show()

    # df.drop(['price_pct_change'],axis1,inplace=True)


