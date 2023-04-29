import keras
import pandas as pd
from matplotlib import pyplot as plt
from src.EDA import EDA
import seaborn as sb
import numpy as np

sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum exploratoryDataAnalysis Pruning for Dec Tree')

    pass
    from sklearn.datasets import load_breast_cancer
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix,accuracy_score
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    #PRUNING
    np.random.seed(123)
    df = sns.load_dataset('titanic')
    df.dropna(how = 'any',inplace=True)
    df.drop(columns=['sex','embarked','class','who','adult_male','deck','embark_town','alive','alone'])

    # df_clean = eda.get_percent_of_missing_observations(df,)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['survived']),df['survived'],
                                                        test_size=0.2,
                                                        random_state=123)
    clf = DecisionTreeClassifier(random_state=123)
    numerics = ['int16','int32','int64','float16','float32','float64']
    # x_train =


    predict_train = clf.predict(x_train)
    accuracy_train = accuracy_score(y_train, predict_train)


    # cf_matrix = confusion_matrix(y_test,y_pred)
    # sns.heatmap(cf_matrix,annot=True,cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()
    # Confusion matrix


    pass
