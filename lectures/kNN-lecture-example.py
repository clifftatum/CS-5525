# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

# For plotting the classification results
from mlxtend.plotting import plot_decision_regions
import warnings
warnings.filterwarnings("ignore")

# Importing the dataset
dataset = load_breast_cancer()

# Converting to pandas DataFrame
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
df['target'] = pd.Series(dataset.target)
print(df.head())

# Selecting the features
features = ['mean perimeter', 'mean texture']
x = df[features]

# Target Variable
y = df['target']

# Splitting the dataset into the training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 25 )


# Fitting KNN Classifier to the Training set
model = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
model.fit(x_train, y_train)

# Predicting the results
y_pred = model.predict(x_test)

# Confusion matrix
print("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# Classification Report
print("\nClassification Report")
report = classification_report(y_test, y_pred)
print(report)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('KNN Classification Accuracy of the model: {:.2f}%'.format(accuracy*100))


# Plotting the decision boundary
plot_decision_regions(x_test.values, y_test.values, clf = model, legend = 2)
plt.title("Decision boundary using KNN Classification (Test)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture")
plt.show()




y_pred_proba = model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=model.classes_ )
disp.plot()
plt.show()


#=========================
# Grid Search for best K
#==========================
knn = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=True, verbose=1)

# fitting the model for grid search
grid_search = grid.fit(x_train, y_train)
print(f'The best k is :  {grid_search.best_params_}')



# viii) Checking Accuracy on Test Data
nn = KNeighborsClassifier(n_neighbors=24)

nn.fit(x_train, y_train)

y_pred_g=nn.predict(x_test)

test_accuracy=accuracy_score(y_test,y_pred_g)*100
matrix = confusion_matrix(y_test, y_pred_g)


#Plot confusion matrix

disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=nn.classes_ )
disp.plot()
plt.show()


print(f"Accuracy before grid search {100*metrics.accuracy_score(y_test, y_pred):.4f}%")
print(f"Accuracy after grid search {100*metrics.accuracy_score(y_test, y_pred_g):.4f}%")
print(f"Precision before grid search {100*metrics.precision_score(y_test, y_pred):.4f}%")
print(f"Precision after grid search {100*metrics.precision_score(y_test, y_pred_g):.4f}%")
print(f"Recall before grid search {100*metrics.recall_score(y_test, y_pred):.4f}%")
print(f"Recall after grid search {100*metrics.recall_score(y_test, y_pred_g):.4f}%")


#==========================================================
# Elbow Method unsupervised kmean method for clustering
#=========================================================
from sklearn.cluster import KMeans
N = 10
wcss = []
for i in range(1, N):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x_train)
    wcss.append(kmeans.inertia_)
plt.plot(np.arange(1,N,1),np.array(wcss), lw=3)
plt.xticks(np.arange(1,N,1))
plt.title('Optimum k in knn method Elbow method ')
plt.ylabel('wcss')
plt.xlabel('number of clusters')
plt.grid()
plt.tight_layout()
plt.show()


model = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
model.fit(x_train, y_train)

# Predicting the results
y_pred_e = model.predict(x_test)



print(f"Accuracy before grid search {100*metrics.accuracy_score(y_test, y_pred):.4f}%")
print(f"Accuracy after grid search {100*metrics.accuracy_score(y_test, y_pred_g):.4f}%")
print(f"Accuracy with elbow {100*metrics.accuracy_score(y_test, y_pred_e):.4f}%")

print(f"Precision before grid search {100*metrics.precision_score(y_test, y_pred):.4f}%")
print(f"Precision after grid search {100*metrics.precision_score(y_test, y_pred_g):.4f}%")
print(f"Precision with elbow {100*metrics.precision_score(y_test, y_pred_e):.4f}%")

print(f"Recall before grid search {100*metrics.recall_score(y_test, y_pred):.4f}%")
print(f"Recall after grid search {100*metrics.recall_score(y_test, y_pred_g):.4f}%")
print(f"Recall with elbow {100*metrics.recall_score(y_test, y_pred_e):.4f}%")