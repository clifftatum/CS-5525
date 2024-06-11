import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_classification(n_samples=1000,
                           n_features=2,
                           n_clusters_per_class=2,
                           n_informative=2,
                           n_repeated=0,
                           n_redundant=0,
                           random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = 0)
x1 = pd.DataFrame(X_train, columns=['X1','X2'])
y1 = pd.DataFrame(y_train, columns=['y1'])
df = pd.concat([x1,y1], axis=1)

plt.figure()
sns.scatterplot(data = df,
                x = df['X1'],
                y = df['X2'],
                hue = 'y1')
plt.grid()
plt.show()
#====================================
# Logistic Regression Classification
#====================================
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

#==================
# ROC & AUC Curve
#==================
from sklearn.metrics import roc_curve, auc
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred)
auc_logitic = auc(logistic_fpr, logistic_tpr)
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, marker = '.', label = f'Logistic regression (auc = {auc:.3f})')
plt.xlabel('False Positive Rate-->')
plt.ylabel('True Positive Rate-->')
plt.legend()
plt.show()
#================================
# Plotting the decision boundary
#===============================
import numpy as np
b = logreg.intercept_[0]
w1, w2 = logreg.coef_.T
c = -b/w2
m = -w1/w2
xmin, xmax = -4, 4
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
sns.scatterplot(data =df,
                x = df['X1'],
                y = df['X2'],
                hue = 'y1')
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')
plt.show()