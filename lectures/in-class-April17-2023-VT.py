from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import numpy as np


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict







X, y = make_classification(n_samples=500,
                           n_features=10,
                           n_informative=5,
                           n_redundant=5,
                           n_classes=4,
                           random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = 0)
Multiclass_model = LogisticRegression(multi_class='ovr')
#fit model
Multiclass_model.fit(X_train, y_train)
y_pred = Multiclass_model.predict(X_test)
probs_y=Multiclass_model.predict_proba(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))

plt.title('Logistic Regression Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()



macro_averaged_precision = metrics.precision_score(y_test, y_pred, average = 'macro')
print(f"Logistic Regression Macro-Averaged Precision score "
      f"using sklearn library : {macro_averaged_precision:.2f}")

micro_averaged_precision = metrics.precision_score(y_test, y_pred, average = 'micro')
print(f"Logistic Regression Micro-Averaged Precision score "
      f"using sklearn library : {micro_averaged_precision:.2f}")

macro_averaged_recall = metrics.recall_score(y_test, y_pred, average = 'macro')
print(f"Logistic Regression Macro-averaged recall score"
      f" using sklearn : {macro_averaged_recall:.2f}")


micro_averaged_recall = metrics.recall_score(y_test, y_pred, average = 'micro')
print(f"Logistic Regression Micro-Averaged recall score "
      f"using sklearn library : {micro_averaged_recall:.2f}")

macro_averaged_f1 = metrics.f1_score(y_test, y_pred, average = 'macro')
print(f"Logistic Regression Macro-Averaged F1 score using sklearn library : {macro_averaged_f1:.2f}")
micro_averaged_f1 = metrics.f1_score(y_test, y_pred, average = 'micro')
print(f"Logistic Regression Micro-Averaged F1 score using sklearn library : {micro_averaged_f1:.2}")
from sklearn.metrics import roc_auc_score
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)
from scipy import mean
print((roc_auc_dict))
print(f'Logistic Regression The mean of AUC for this multi-label classificationin is '
      f'{100*mean(list(roc_auc_dict.values())):.4f}%')
# Importing the dataset
# dataset = load_breast_cancer()
# # Converting to pandas DataFrame
# df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
# df['target'] = pd.Series(dataset.target)
# print(df.head())
#
# # Selecting the features
# features = ['mean perimeter', 'mean texture']
# x = df[features]
# x = (x-x.mean())/x.std()
# y = df['target']
#
# # Splitting the dataset into the training and test set
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,
#                                                     shuffle=True,
#                                                     random_state = 25 )
# model = SVC(kernel='linear',random_state = 0, probability=True)
# model.fit(x_train, y_train)
# # Predicting the results
# y_pred = model.predict(x_test)
# y_pred_pr = model.predict_proba(x_test)
# # Confusion matrix
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# print("Confusion Matrix")
# matrix = confusion_matrix(y_test, y_pred)
# print(matrix)
#
# print("\nClassification Report")
# report = classification_report(y_test, y_pred)
# print(report)
#
# # Accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print('SVM Classification Accuracy of the model: {:.2f}%'.format(accuracy*100))
#
# from mlxtend.plotting import plot_decision_regions
# # Plotting the decision boundary
# plot_decision_regions(x_test.values, y_test.values, clf = model, legend = 2)
# plt.title("Decision boundary using SVC (Test)")
# plt.xlabel("mean_perimeter")
# plt.ylabel("mean_texture")
# plt.show()
#
# #====================
# # Plot ROC and AUC
# #===================
# from sklearn.metrics import roc_curve, auc
# fpr, tpr, threshold = roc_curve(y_test, y_pred_pr[:,1])
# auc = auc(fpr, tpr)
#
# plt.figure(figsize=(5,5), dpi = 100)
# plt.plot(fpr, tpr,linestyle = '--', label = f'Logistic (auc = {auc:.3f})')
# plt.xlabel('False Positive Rate-->')
# plt.ylabel('True Positive Rate-->')
# plt.legend()
# plt.show()
#
