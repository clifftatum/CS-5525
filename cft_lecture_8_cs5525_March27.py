import pandas as pd
from matplotlib import pyplot as plt
from src.EDA import EDA
import seaborn as sb
import numpy as np
import statsmodels.api as sm
np.random.seed(5525)

sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lecture 8 - March 27 CS-5525')
    # Apply Logistic Regression to Some Random Data Set
    from sklearn.datasets import make_regression, make_classification, make_blobs
    from sklearn.model_selection import train_test_split

    X,y = make_classification(n_clusters_per_class=2,
                              n_informative=2,
                              n_samples=1000,
                              n_repeated=0,
                              n_redundant=0,
                              random_state=123,
                              n_features=2)

    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=0)
    df = pd.DataFrame(np.hstack((x_train,y_train.reshape(len(y_train),1))),columns=['x1','x2','y'])

    def change(x):
        thresh = 0.5

        if x> thresh:
            return '1'
        else:
            return'0'

    df['y1'] = df['y'].apply(change)


    import plotly.express as px
    fig = px.scatter(df, x="x1", y="x2", color="y1")
    # fig.show()

    # Logistic Regress Class
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    y_pred = logreg.predict(x_test)
    cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
    print(cnf_matrix)

    #Roc and Auc
    from sklearn.metrics import roc_curve, auc
    logistic_fpr, logistic_tpr, threshhold = roc_curve(y_test,y_pred)
    auc_logistic = auc(logistic_fpr,logistic_tpr)
    y_pred_prob = logreg.predict_proba(x_test)[::,1]
    fpr,tpr,threshold = roc_curve(y_test,y_pred_prob)
    auc = auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,marker='.',label = str('Logistic Regression auc= '+auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # z = b+w!x1+w2x2 = 0 therefore slope = -w1/w2,bias = -b/w2
    b = logreg.intercept_[0]
    w1,w2 = logreg.coef_.T
    c = -b/w2
    m = -w1/w2
    xmin,xmax = -4,4
    xd = np.array([xmin,xmax])

    plt.plot(xd,yd,'k',lw=1,ls='--')
    # plot the same scatterplot as above too

    pass
