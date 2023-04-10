import pandas as pd
from matplotlib import pyplot as plt
from src.EDA import EDA
import seaborn as sb
import plotly.express as px
import numpy as np

sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format
np.random.seed(5525)

if __name__ == '__main__':
    eda = EDA('Clifford Tatum HW 4 CS-5525')
    show_plot = False

    # Problem 1
    prob_p1 = []
    ginis = []
    entropies = []
    for x in range(10000):
        p1_temp = np.random.uniform(0, 4)
        p2_temp = abs(4 - p1_temp)
        gini_temp = 1 - ((p1_temp / (p1_temp + p2_temp)) ** 2 + (p2_temp / (p1_temp + p2_temp)) ** 2)
        try:
            entropy_temp = (-(p1_temp / (p1_temp + p2_temp)) * np.log2((p1_temp / (p1_temp + p2_temp))) -
                            (p2_temp / (p1_temp + p2_temp)) * np.log2((p2_temp / (p1_temp + p2_temp))))
        except ValueError:
            entropy_temp = 0

        ginis.append(gini_temp)
        entropies.append(entropy_temp)
        prob_p1.append((p1_temp / (p1_temp + p2_temp)))

    df_e_g = pd.DataFrame(np.hstack([np.array(prob_p1).reshape(len(prob_p1), 1),
                                     np.array(entropies).reshape(len(entropies), 1),
                                     np.array(ginis).reshape(len(ginis), 1)]),
                          columns=['Probability', 'Entropy', 'Gini'])
    fig1 = px.scatter(df_e_g, x='Probability', y=['Entropy', 'Gini'])
    fig1.update_layout(plot_bgcolor='white')
    fig1.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig1.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig1.update_yaxes(tickfont_family="Arial Black")
    fig1.update_xaxes(tickfont_family="Arial Black")
    fig1.update_xaxes(showgrid=True)
    fig1.update_yaxes(showgrid=True)
    fig1.update_layout(font=dict(size=18), yaxis_title='<b>Magnitude<b>', xaxis_title='<b>P<b>',
                       title_text='<b>Entropy versus Gini index<b>')
    if show_plot:
        fig1.show()

    # Problem 2
    z = np.linspace(-10, 10, 100)
    sigma_of_z = [1 / (1 + np.exp(-Z)) for Z in z]
    cost = []
    cost_yeq0 = []
    N = 2  # Binary logg loss function ( 0 or 1 )
    for s_z in sigma_of_z:
        # y_pred = np.clip(s_z, 1e-15, 1 - 1e-15)
        y_pred = s_z
        cross_entropy_1_true = (1 / N) * (-np.sum(1 * np.log(y_pred) + (1 - 1) * np.log(1 - y_pred)))
        cross_entropy_0_true = (1 / N) * (-np.sum(0 * np.log(y_pred) + (1 - 0) * np.log(1 - y_pred)))
        cost.append(cross_entropy_1_true)
        cost_yeq0.append(cross_entropy_0_true)

    df_log_loss = pd.DataFrame(np.hstack([np.array(sigma_of_z).reshape(len(sigma_of_z), 1),
                                          np.array(cost_yeq0).reshape(len(cost_yeq0), 1),
                                          np.array(cost).reshape(len(cost), 1)]),
                               columns=['sigma_of_z', 'J(w) if y=0', 'J(w) if y=1'])
    import plotly.graph_objects as go

    fig2 = go.Figure()
    t1 = fig2.add_trace(go.Scatter(name='J(w) if y=1', x=df_log_loss['sigma_of_z'], y=df_log_loss['J(w) if y=1']))
    t2 = fig2.add_trace(go.Scatter(name='J(w) if y=0', x=df_log_loss['sigma_of_z'],
                                   y=df_log_loss['J(w) if y=0'],
                                   line=dict(dash='dash')))

    # fig2 = px.line(df_log_loss, x='sigma_of_z', y=['J(w) if y=1','J(w) if y=0'])
    fig2.update_layout(plot_bgcolor='white')
    fig2.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig2.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig2.update_yaxes(tickfont_family="Arial Black")
    fig2.update_xaxes(tickfont_family="Arial Black")
    fig2.update_xaxes(showgrid=True)
    fig2.update_yaxes(showgrid=True)
    fig2.update_traces(line=dict(width=10))
    fig2.update_layout(font=dict(size=25), yaxis_title='J(w)', xaxis_title='<b>$ \sigma (z) $<b>',
                       title_text='Log-loss function')
    if show_plot:
        fig2.show()

    # Problem 5
    import seaborn as sns

    df = sns.load_dataset('titanic')
    # Get numerical features
    df = eda.get_numerical_features(df.copy(deep=True))
    # Clean using means
    _, _, df = eda.get_percent_of_missing_observations(df.copy(deep=True), clean=True, show=False, clean_method='mean')
    # Split the data
    x_train, x_test, y_train, y_test = eda.split_80_20(df.copy(deep=True), target='survived')
    # Train the classifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    clf = DecisionTreeClassifier(random_state=123)
    clf.fit(x_train, y_train)

    # Show the feature importance
    importance = clf.feature_importances_
    indices = np.argsort(importance)[-int(len(x_train.columns)):]  # top 20
    sorted_important_features = [x_train.columns[i] for i in indices]
    importances = dict(zip(sorted_important_features, np.round(importance, 2)))
    print(f'Decision Tree Classifier Feature importances : {importances}')

    # Performance Metrics
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    y_hat = clf.predict(x_test)
    print('Generic Model - Decision Tree Classifier')
    print(f"Accuracy score against test dataset: {accuracy_score(y_test, y_hat):.2f}")
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_hat, average='binary')
    print(f"Precision score against test dataset: {precision:.2f}")
    print(f"Recall score against test dataset:{recall:.2f}")
    print(f"F-Score score against test dataset:{fscore:.2f}")

    # Problem 6
    _, best_params, model = eda.grid_search_2D(model_type='Decision Tree Classifier',
                                               X=x_train, y=y_train)
    print(f"Optimum parameters for Decision Tree Classifer grid search:{best_params}")

    DTC_model, results_DTC, _ = eda.drop_and_show_model_prediction(fitted_model=model.fit(x_train, y_train),
                                                                   x_train=x_train.copy(deep=True),
                                                                   y_train=y_train.copy(deep=True),
                                                                   x_test=x_test.copy(deep=True),
                                                                   y_test=y_test.copy(deep=True),
                                                                   dropped_feats=None,
                                                                   show=True,
                                                                   title=None,
                                                                   compute_prediction=True,
                                                                   compute_method='package',
                                                                   dim_red_method=None)

    # Problem 7
    y_hat = results_DTC['Predicted survived'].values
    print('Best Grid Model - Decision Tree Classifier')
    print(f"Accuracy score against test dataset: {accuracy_score(y_test, y_hat):.2f}")
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_hat, average='binary')
    print(f"Precision score against test dataset: {precision:.2f}")
    print(f"Recall score against test dataset:{recall:.2f}")
    print(f"F-Score score against test dataset:{fscore:.2f}")

    # Problem 9
    from sklearn.linear_model import LogisticRegression
    log_reg_model = LogisticRegression(max_iter=2500,
                                       random_state=123)
    log_reg_model.fit(x_train, y_train)
    # Make predictions
    y_hat = log_reg_model.predict(x_test)
    # Show results
    print('Model - Logistic Regression')
    print(f"Accuracy score against test dataset: {accuracy_score(y_test, y_hat):.2f}")
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_hat, average='binary')
    print(f"Precision score against test dataset: {precision:.2f}")
    print(f"Recall score against test dataset:{recall:.2f}")
    print(f"F-Score score against test dataset:{fscore:.2f}")


    pass
