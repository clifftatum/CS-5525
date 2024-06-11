import pandas as pd
from src.data_analytics import ExploratoryDataAnalysis, RegressionAnalysis, \
    ClassificationAnalysis, ClusteringAnalysis, AssociationMining
import numpy as np
import statsmodels.api as sm

np.random.seed(5525)

pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = ExploratoryDataAnalysis(
        'Clifford Tatum ExploratoryDataAnalysis Deflation Market Nov 2022 - Jan 2023 - CS-5525')
    ra = RegressionAnalysis()
    am = AssociationMining()
    ca = ClassificationAnalysis()
    #
    df_init = pd.read_csv('C:\\Users\\cft5385\\'
                     'Documents\\Learning\\'
                     'CS-5525\\Code\\Term Project\\'
                     'CS-5525-Term-Project\\DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv')

    # df_init = pd.read_csv('C:\\Users\\cft5385\\'
    #                       'Documents\\Learning\\'
    #                       'CS-5525\\pythonProject\\'
    #                       'CS-5525-Term-Project\\DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv')

    fig_pack = []
    truncate = False
    show = False
    if truncate:
        df_init = df_init.iloc[1:100, :]
    ####################################################################################################################
    # Phase I: EDA
    ####################################################################################################################

    # Change the date feature to a datetime
    df_init['Date'] = pd.to_datetime(df_init['Date'])

    # Sort the date by oldest to newest
    df_init.sort_values(by='Date', inplace=True, ascending=True)

    # Delete the useless/redundant features(s)
    df_init.drop(columns=['Unnamed: 0', 'target_signal_performance'], inplace=True)
    num_missing, percent_removed, df_init = eda.get_percent_of_missing_observations(df_init.copy(deep=True),
                                                                                    clean=True,
                                                                                    show=True,
                                                                                    clean_method='prune')


    # Make some of the feature's Names more detailed rather than just acronyms
    def change_s_type(x):
        if x == 'E':
            return 'ETF'
        elif x == 'U':
            return 'Currency'
        elif x == 'F':
            return 'FixedIncome'
        elif x == 'M':
            return 'Commodity'
        elif x == 'S':
            return 'StockEquity'
        elif x == 'I':
            return 'Index'


    df_init['StockType'] = df_init['StockType'].apply(change_s_type)


    def change_ssc(x):
        if x == 'T':
            return 'Topping'
        elif x == 'B':
            return 'Bottoming'
        elif x == 'S':
            return 'Sell'
        elif x == 'Y':
            return 'Buy'
        elif x == 'I':
            return 'Intermediate'


    df_init['SSCrossover'] = df_init['SSCrossover'].apply(change_ssc)


    def change_e(x):
        if x == 'R':
            return 'Press'
        elif x == 'O':
            return 'Short'
        elif x == 'I':
            return 'Intermediate'


    df_init['SSEmbedded'] = df_init['SSEmbedded'].apply(change_e)


    def change_f(x):
        if x == 'K':
            return 'Breakout'
        elif x == 'X':
            return 'Exit'
        elif x == 'I':
            return 'Intermediate'


    df_init['SSFailure'] = df_init['SSFailure'].apply(change_f)

    # Print the target name:
    target_name = df_init.columns[-1]
    print(f'Target: {target_name}')


    # Count Plot of the Target to show the label distribution

    # For now, I'm electing to not Aggregate given that it is computational manageable

    # I am electing to Down-sample the Categorical Feature: 'Industry' which has 327 unique categories.
    # Because the chosen encoding method for converting categorical data to numerical data is one-hot encoding,
    # which increases the dimensionality of the data I elect to remove Industry as a feature,
    # but will keep it with my data set

    # Identify the columns which are not features but should remain with the dataset
    not_features = ['Date', 'Industry', 'Symbol', target_name]

    # One-hot-encoding converts the qualitative features to quantitative feature
    df_encoded, _, encoded_feats = eda.one_hot_encode(df=df_init.copy(deep=True),
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
    keep_ind, perc_detected_removed,_ = eda.detect_and_remove_outliers(
        df=df_encoded.copy(deep=True)[feats_to_standardize],
        method='z_score')

    # Remove the outliers
    df_encoded = df_encoded.iloc[keep_ind.T, :]

    fig0 = eda.show_aggregate(df=df_encoded[feats_to_standardize], df_agg=None,
                              plot_type='plotly',
                              title='<b> Numerical Feature Distribution after Outlier Detection and Removal <b>',
                              split=['Volume', 'VolumeTenMA'])
    fig_pack.append(fig0)


    # Show the classification target label distribution
    fig_count = eda.show_count(df=df_encoded,target=target_name)
    # fig_count.show()
    fig_pack.append(fig_count)




    # Standardize (not normalize) the appropriate data
    df_encoded[feats_to_standardize] = eda.standardize(df=df_encoded[feats_to_standardize].copy(deep=True))

    # I should standardize the data, since normalization is not appropriate when handleing numerical features that have
    # significantly different scales or units. Standardization will scale the data to have mean 0 and standard
    # deviation 1, which will allow you to compare and combine the variables on a common scale

    df = df_encoded.iloc[:, len(non_feats):]
    df.drop(columns = ['StockType_Index'],inplace=True)

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


    # Both PCA and SVD show that only 7/45 features are required for 90% explained variance,
    # however, 20/45 features provides 98% explained variance, move forward with 20 as the keep threshold

    # Random Forest Analysis (RFA)
    fig3, drop_these_rfa_class, n_most_import_class = eda.random_forest_analysis(X=df.copy(deep=True).drop(columns=target_name),
                                                                         y=df.copy(deep=True)[target_name],
                                                                         max_features=7,
                                                                         plot_type='plotly',
                                                                         title='<b>Random Forest Analysis<b>',
                                                                         n_most_important = 3)
    fig_pack.append(fig3)

    # Show the Covariance and Pearson Correlation Coefficient Matrices
    # fig4 = eda.show_cov(df=df.copy(deep=True).drop(columns=target_name))
    # fig_pack.append(fig4)
    #
    # fig5 = eda.show_corr(df=df.copy(deep=True).drop(columns=target_name))
    # fig_pack.append(fig5)

    # # Split the Dataset to train and test (80%-20%) with Shuffle=False
    # x_train,x_test,y_train,y_test = eda.split_80_20(df=df.copy(deep=True),
    #                                                 target=target_name)

    ####################################################################################################################
    # Phase II: Regression Analysis
    ####################################################################################################################

    # Final Regression Analysis and Confidence Interval Analysis for Dependant Variable: Bollinger BandWidth.
    # Comparing the 2 linear regression models formulated by diminished feature sets
    # of dimensionailty reduction methods: RFA and BLR


    # Random Forest Analysis (RFA) and Backward Linear Regression (BLR)
    # Get a new Train/Test Dataset with Target='BandWidth'
    reg_target = 'PctB'
    df[reg_target] = df_init[reg_target]


    # Random Forest Analysis (RFA)
    fig_rfa, drop_these_rfa, n_most_import_reg = eda.random_forest_analysis(X=df.copy(deep=True).drop(columns=[reg_target, target_name]),
                                                                         y=df.copy(deep=True)[reg_target],
                                                                         max_features=7,
                                                                         plot_type='plotly',
                                                                         title='<b>Random Forest Analysis<b>',
                                                                         n_most_important = 7)
    fig_pack.append(fig_rfa)

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = \
        eda.split_80_20(df=df.copy(deep=True).drop(columns=['target_buy_sell_performance']), target=reg_target)

    # BLR Results
    linear_reg_model_blr,\
    results_lin_reg_blr,\
    fig_blr\
        ,drop_these_blr = ra.backward_linear_regression(  x_train=sm.add_constant(x_train_reg.copy(deep=True)),
                                                          y_train=y_train_reg.copy(deep=True),
                                                          x_test=x_test_reg.copy(deep=True),
                                                          y_test=y_test_reg.copy(deep=True),
                                                          compute_prediction=True,
                                                          compute_method='package',
                                                          show=True,
                                                          encode_target=False,
                                                          req_num_feats =7 )
    fig_pack.append(fig_blr)


    # RFA Results
    linear_reg_model_rfa,\
    linear_reg_results_rfa,fig_rfa2\
        = ra.drop_and_show_regression_results(x_train=sm.add_constant(x_train_reg.copy(deep=True)),
                                                                  y_train=y_train_reg.copy(deep=True),
                                                                  x_test=x_test_reg.copy(deep=True),
                                                                  y_test=y_test_reg.copy(deep=True),
                                                                  dropped_feats=drop_these_rfa,
                                                                  show=True,
                                                                  title='Random Forest Analysis Resultant Feature Space Regression Results',
                                                                  compute_prediction=True,
                                                                  compute_method='package',
                                                                  dim_red_method='Random Forest Analysis')
    fig_pack.append(fig_rfa2)


    # Compare BLR v RFA

    # Yes, a small confidence interval indicates that you are highly confident that the true population parameter lies
    # within a narrow range of values. However, keep in mind that the size of the confidence interval
    # depends on several factors, such as the sample size, the level of significance, and the variability of the data.

    df_final_res, figs, df_predictions = ra.compare_regression_models(model_a=linear_reg_model_blr,
                                                                      model_b=linear_reg_model_rfa,
                                                                      mod_a_distinct_method='Final Regression Model: '
                                                                                            'Backward Linear Regression',
                                                                      mod_b_distinct_method='Random Forest Analysis',
                                                                      mod_a_res=results_lin_reg_blr,
                                                                      mod_b_res=linear_reg_results_rfa,
                                                                      show=True, target_name=reg_target)

    for f in figs:
        fig_pack.append(f)

     # Based on the regression results beween RFA and BLR, BLR is selected as the final regression model



    # Print the dimensionality reduction results for each of the following:
    # Resultant feature set via Random Forest Analysis to determine important features for classification
    # Resultant feature set via Random Forest Analysis to determine important features in regression
    # Resultant feature set via Stepwise Regression Analysis to determine important features in regression
    print(f'Resultant feature set via Random Forest Analysis to determine important features for classification'
          f' {list(set(df.copy(deep=True).drop(columns=target_name).columns) -set(drop_these_rfa_class))}')

    print(f'Resultant feature set via Random Forest Analysis to determine important features in regression'
          f' {list(set(x_train_reg.columns) -set(drop_these_rfa))}')

    print(f'Resultant feature set via Stepwise Regression Analysis to determine important features in regression'
          f' {list(set(x_train_reg.columns) -set(drop_these_blr))}')


    # For the remaining FTP analysis use the union of all 3 above features space result sets
    drop_set_final = list(set(df.copy(deep=True).drop(columns=target_name).columns) -\
                     set(np.unique(list(set(df.copy(deep=True).drop(columns=target_name).columns)
                                    - set(drop_these_rfa_class))
                               +list(set(x_train_reg.columns) - set(drop_these_rfa))
                               +list(set(x_train_reg.columns) -set(drop_these_blr)))))
    drop_set_final.append('Open')

    x_train, x_test, y_train, y_test = \
        eda.split_80_20(df=df.copy(deep=True).drop(columns=drop_set_final), target=target_name,
                        stratify =np.unique(df.copy(deep=True)[target_name]))

    drop_set_final_reg =drop_set_final
    drop_set_final_reg.remove(reg_target)
    drop_set_final_reg.append(target_name)

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = \
        eda.split_80_20(df=df.copy(deep=True).drop(columns=drop_set_final_reg), target=reg_target)



    # Get the ANOVA / t-test and f-test for the model
    # I have a categorical response variable (y), therefore the t-test and f-test are not appropriate.
    # These tests assume a continuous response variable, and do not work well with categorical data.
    # The ANOVA test (Analysis of Variance) can be used to test whether there are significant differences between
    # the means of two or more groups. Therefore, I can test whether there are significant differences
    # in the mean values of a continuous predictor variable across different levels of a categorical response variable.

    # The t-test is used to test whether there is a significant difference between the means of two groups.
    # It calculates a t-statistic that measures the difference between the means of the two groups relative
    # to the variation within each group. A small p-value (typically less than 0.05) indicates that there is
    # a significant difference between the two groups.
    # The f-test, on the other hand, is used to test whether there is a significant difference between the means of two
    # or more groups. It calculates an F-statistic that compares the variation between the groups to the variation
    # within each group. A small p-value (typically less than 0.05) indicates that there is a
    # significant difference between the means of the groups.

    t_test_f_test_res = eda.test_null_hypothosis(X=x_train_reg.copy(deep=True),
                                                 y=y_train_reg.copy(deep=True),
                                                 method='t_test')

    # Association Analysis - for now use the initial categorical features fod the association analysis
    # df_assoc = df_encoded[encoded_feats]
    df_assoc_encoded = pd.DataFrame(columns=x_train_reg.columns)
    # df_assoc_encoded.iloc[:,5:] = x_train_reg.iloc[:,5:]

    # discretize / bin the numerical features
    for feat in df_assoc_encoded.iloc[:,:5].columns:
        df_assoc_encoded[feat] = pd.cut(x=x_train_reg[feat].astype(float),
                                   bins=[np.min(x_train_reg[feat]),
                                         np.mean(x_train_reg[feat]),
                                         np.max(x_train_reg[feat])],
                                   labels=["Low", "High"])
        df_assoc_encoded[feat] = df_assoc_encoded[feat].fillna(df_assoc_encoded[feat].mode()[0])

    # One-hot-encoding converts the qualitative features to quantitative feature
    df_encoded_temp, _, _ = eda.one_hot_encode(df=df_assoc_encoded.iloc[:,:5].copy(deep=True),
                                               not_features=None)
    d_encod_names =df_encoded_temp.columns
    other_names = x_train_reg.iloc[:,5:].columns
    df_for_assoc = pd.DataFrame(np.hstack((df_encoded_temp.values,x_train_reg.iloc[:,5:].values)),
                                columns=list(d_encod_names) + list(other_names))

    df_assoc_res = am.get_associations(df=df_for_assoc, show=True, method='apriori')

    # Collinearity Analysis
    df_VIF = ra.get_collinearity(X=x_train_reg, method='VIF')
    # The results of Collinearity Analysis make sense, Price specific features, like Open, High, Low, Moving Average are
    # very co-linear.

    # Now show the corr and cov matrices for the dimensionally reduced feature space
    fig_cov_final = eda.show_cov(df=x_train)
    fig_pack.append(fig_cov_final)

    fig_corr_final = eda.show_corr(df=x_train)
    fig_pack.append(fig_corr_final)


    ####################################################################################################################
    # Phase III: Classification
    ####################################################################################################################
    # for f in fig_pack:
    #     f.show()

    ##################################
    #  Decision Tree Classifier
    ##################################

    # Create a 'Full' Decision Tree Classifier - no pruning
    full_dtc = ca.get_decision_tree_classifier(x_train=x_train.copy(deep=True),
                                               y_train=y_train.copy(deep=True),
                                               fit=False)

    # Convert the Full Decision Tree Classifier Model to a One Versus All multi-label classifier
    full_dtc_ova_fitted = ca.to_one_vs_all(full_dtc, x_train=x_train.copy(deep=True),
                                           y_train=y_train.copy(deep=True),
                                           fit=True)

    # Convert the Full Decision Tree Classifier Model to a One Versus One multi-label classifier
    full_dtc_ovo_fitted = ca.to_one_vs_one(full_dtc, x_train=x_train.copy(deep=True),
                                           y_train=y_train.copy(deep=True),
                                           fit=True)

    ##################################
    # Random Forest Tree
    ##################################
    rfc = ca.get_random_forest_classifier(x_train=x_train.copy(deep=True),
                                          y_train=y_train.copy(deep=True),
                                          fit=False)

    rfc_ovo_fitted = ca.to_one_vs_one(model=rfc,
                                      x_train=x_train.copy(deep=True),
                                      y_train=y_train.copy(deep=True),
                                      fit=True)

    rfc_ova_fitted = ca.to_one_vs_all(model=rfc,
                                      x_train=x_train.copy(deep=True),
                                      y_train=y_train.copy(deep=True),
                                      fit=True)
    ##################################
    #  Logistic Classifier
    ##################################

    lrc = ca.get_logistic_regression_classifier(x_train=x_train.copy(deep=True),
                                               y_train=y_train.copy(deep=True),
                                               fit=False)

    lrc_ova_fitted = ca.to_one_vs_all(lrc, x_train=x_train.copy(deep=True),
                                           y_train=y_train.copy(deep=True),
                                           fit=True)

    lrc_ovo_fitted = ca.to_one_vs_one(lrc, x_train=x_train.copy(deep=True),
                                           y_train=y_train.copy(deep=True),
                                           fit=True)

    ##################################
    #  Naive Bayes Classifier
    ##################################

    gnb = ca.get_naive_bayes(x_train=x_train.copy(deep=True),
                             y_train=y_train.copy(deep=True),
                             fit=False)

    gnb_ova_fitted = ca.to_one_vs_all(gnb, x_train=x_train.copy(deep=True),
                                           y_train=y_train.copy(deep=True),
                                           fit=True)

    gnb_ovo_fitted = ca.to_one_vs_one(gnb, x_train=x_train.copy(deep=True),
                                           y_train=y_train.copy(deep=True),
                                           fit=True)

    ##################################
    #  Support Vector Machine Classifier
    ##################################

    svm = ca.get_SVM(x_train=x_train.copy(deep=True),
                     y_train=y_train.copy(deep=True),
                     fit=False)

    svm_ova_fitted = ca.to_one_vs_all(svm, x_train=x_train.copy(deep=True),
                                           y_train=y_train.copy(deep=True),
                                           fit=True)

    svm_ovo_fitted = ca.to_one_vs_one(svm, x_train=x_train.copy(deep=True),
                                           y_train=y_train.copy(deep=True),
                                           fit=True)






    ####################################################
    # Make Predictions for each of the classifiers
    ####################################################

    # Make a prediction using the full tree  - One Versus all Classification
    results_dtc_full_tree_ova, _ = ca.predict_fitted_model(fitted_model=full_dtc_ova_fitted,
                                                           x_test=x_test.copy(deep=True),
                                                           y_test=y_test.copy(deep=True),
                                                           compute_prediction=True,
                                                           show=False,
                                                           title=None)

    # Make a prediction using the full tree  - One Versus One Classification
    results_dtc_full_tree_ovo, _ = ca.predict_fitted_model(fitted_model=full_dtc_ovo_fitted,
                                                           x_test=x_test.copy(deep=True),
                                                           y_test=y_test.copy(deep=True),
                                                           compute_prediction=True,
                                                           show=False,
                                                           title=None)

    # Make a prediction using the Random Forest Classifier - One Versus all Classification
    results_rfc_ova, _ = ca.predict_fitted_model(fitted_model=rfc_ova_fitted,
                                                 x_test=x_test.copy(deep=True),
                                                 y_test=y_test.copy(deep=True),
                                                 compute_prediction=True,
                                                 show=False,
                                                 title=None)

    # Make a prediction using the Random Forest Classifier  - One Versus One Classification
    results_rfc_ovo, _ = ca.predict_fitted_model(fitted_model=rfc_ovo_fitted,
                                                 x_test=x_test.copy(deep=True),
                                                 y_test=y_test.copy(deep=True),
                                                 compute_prediction=True,
                                                 show=False,
                                                 title=None)

    # Make a prediction using the Logistic Regression Classifier - One Versus all Classification
    results_lrc_ova, _ = ca.predict_fitted_model(fitted_model=lrc_ova_fitted,
                                                 x_test=x_test.copy(deep=True),
                                                 y_test=y_test.copy(deep=True),
                                                 compute_prediction=True,
                                                 show=False,
                                                 title=None)

    # Make a prediction using the Logistic Regression Classifier  - One Versus One Classification
    results_lrc_ovo, _ = ca.predict_fitted_model(fitted_model=lrc_ovo_fitted,
                                                 x_test=x_test.copy(deep=True),
                                                 y_test=y_test.copy(deep=True),
                                                 compute_prediction=True,
                                                 show=False,
                                                 title=None)

    # Make a prediction using the Gaussian Naive Bayes Classifier - One Versus all Classification
    results_gnb_ova, _ = ca.predict_fitted_model(fitted_model=gnb_ova_fitted,
                                                 x_test=x_test.copy(deep=True),
                                                 y_test=y_test.copy(deep=True),
                                                 compute_prediction=True,
                                                 show=False,
                                                 title=None)

    # Make a prediction using the Gaussian Naive Bayes Classifier  - One Versus One Classification
    results_gnb_ovo, _ = ca.predict_fitted_model(fitted_model=gnb_ovo_fitted,
                                                 x_test=x_test.copy(deep=True),
                                                 y_test=y_test.copy(deep=True),
                                                 compute_prediction=True,
                                                 show=False,
                                                 title=None)


    # Make a prediction using the Support Vector Machine Classifier - One Versus all Classification
    results_svm_ova, _ = ca.predict_fitted_model(fitted_model=svm_ova_fitted,
                                                 x_test=x_test.copy(deep=True),
                                                 y_test=y_test.copy(deep=True),
                                                 compute_prediction=True,
                                                 show=False,
                                                 title=None)

    # Make a prediction using the Support Vector Machine Classifier  - One Versus One Classification
    results_svm_ovo, _ = ca.predict_fitted_model(fitted_model=svm_ovo_fitted,
                                                 x_test=x_test.copy(deep=True),
                                                 y_test=y_test.copy(deep=True),
                                                 compute_prediction=True,
                                                 show=False,
                                                 title=None)





    ####################################################
    # Show the performance for each of the classifiers
    ####################################################

    models = [full_dtc_ova_fitted,
              full_dtc_ovo_fitted,
              rfc_ova_fitted,
              rfc_ovo_fitted,
              lrc_ova_fitted,
              lrc_ovo_fitted,
              gnb_ova_fitted,
              gnb_ovo_fitted,
              svm_ova_fitted,
              svm_ovo_fitted,
              ]

    mod_dist_meths = ['Decision Classifier Tree (One vs. All)',
                      'Decision Classifier Tree (One vs. One)',
                      'Random Forest Classifier Tree (One vs. All)',
                      'Random Forest Classifier Tree (One vs. One)',
                      'Logistic Regression Classifier Tree (One vs. All)',
                      'Logistic Regression Classifier Tree (One vs. One)',
                      'Naive Bayes Classifier Tree (One vs. All)',
                      'Naive Bayes Classifier Tree (One vs. One)',
                      'Support Vector Machine Classifier Tree (One vs. All)',
                      'Support Vector Machine Classifier Tree (One vs. One)'
                      ]

    mod_res = [results_dtc_full_tree_ova,
               results_dtc_full_tree_ovo,
               results_rfc_ova,
               results_rfc_ovo,
               results_lrc_ova,
               results_lrc_ovo,
               results_gnb_ova,
               results_gnb_ovo,
               results_svm_ova,
               results_svm_ovo,
               ]

    # Compare the Classifiers # TODO, the metrics are delivered currently for binary lable classifiers, fix this!
    df_ovo,df_ova, figs = ca.show_classification_models(        models=models,
                                                                methods=mod_dist_meths,
                                                                results=mod_res,
                                                                x_train=x_train.copy(deep=True),
                                                                y_train=y_train.copy(deep=True),
                                                                x_test=x_test.copy(deep=True),
                                                                y_test=y_test.copy(deep=True),
                                                                show=True,
                                                                average='macro',
                                                                target_labels = np.unique(y_train))


    ###################################################################################################################
    # Association Analysis
    ###################################################################################################################

    # Kmeans Clustering
    fig_cluster = am.get_k_mean_clusters(x_train,n_clusters=2)
    fig_pack.append(fig_cluster)

    # DBSCAN clustering
    fig_cluster = am.get_dbscan_clusters(x_train)
    fig_pack.append(fig_cluster)






    for f in figs:
        fig_pack.append(f)
        f.show()


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
