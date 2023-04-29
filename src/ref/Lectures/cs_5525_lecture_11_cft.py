import pandas as pd
from matplotlib import pyplot as plt
from src.EDA import EDA
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import tree
from scipy.stats import gmean
from scipy.stats import hmean
np.set_printoptions(precision=3)
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# sb.set(color_codes=True)
# pd.set_option("display.precision", 2)
# pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lecture 11 April 17 2022 - CS-5525')
    ########################
    # Support Vector Machines
    ########################
    import sklearn
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

    from sklearn.svm import SVC
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
    df['target'] = pd.Series(dataset.target)
    print(df.head().to_string())

    features = ['mean perimeter',' mean texture']
    x = df['features']
    y = df['target']
    x = (x-x.mean())/x.std()

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=1,
                                                        shuffle=True)  # False if using temporal data
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0])


    pass
