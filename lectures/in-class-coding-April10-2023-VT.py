import numpy as np
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
# Encoding Features
# First, you need to convert these string labels into numbers. for example:
# 'Overcast', 'Rainy', 'Sunny' as 0, 1, 2.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
weather_encoded=le.fit_transform(weather)
print("weather:", weather_encoded)

temp_encoded=le.fit_transform(temp)# Hot:1....Mild:2......0:Cool
label=le.fit_transform(play)# 0: No.....1: Yes
print("Temp:",temp_encoded)
print("Play:",label)
features=np.vstack((np.array(weather_encoded),np.array(temp_encoded)))
print(features)

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(features.reshape(-1,2),label)
predicted= model.predict([[2,1]]) #
print("Predicted Value:", predicted)
if predicted==0:
    print('Not play')
else:
    print('Will play')



# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
# #=======================================
# # ii. Modeling without Post-Pruning
# #=======================================
# X,y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                             random_state=0,
#                                             test_size=0.2,
#                                             shuffle=True)
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(X_train,y_train)
# y_train_predicted = clf.predict(X_train)
# print(f'Train accuracy {accuracy_score(y_train, y_train_predicted)}')
#
# y_test_predicted = clf.predict(X_test)
# accuracy_score(y_test, y_test_predicted)
# print(f'Test accuracy {accuracy_score(y_test, y_test_predicted)}')
# # As we see diference between accuracy score of train and test is too
# # high means model is overfitted (because it
# # is accurate for training set but gives large error when we provide test set to the model)
#
#
#
# #Visualizing Decision Tree
#
# plt.figure(figsize=(16,8))
# tree.plot_tree(clf,rounded=True,filled=True)
# plt.show()
#
# #======================================
# # iii. Post-Pruning operation :
# #=======================================
# path = clf.cost_complexity_pruning_path(X_train, y_train)
# print(path)
# # Plot
#
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
# print("ccp alpha wil give list of values :",ccp_alphas)
# print("***********************************************************")
# print("Impurities in Decision Tree :",impurities)
#
# plt.figure(figsize=(10, 6))
# plt.plot(ccp_alphas, impurities)
# plt.xlabel("effective alpha")
# plt.ylabel("total impurity of leaves")
# plt.show()
# #====================================================
# # Finding an optimal value of alpha using Python
# #====================================================
# clfs = []
#
# for ccp_alpha in ccp_alphas:
#     clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
#     clf.fit(X_train, y_train)
#     clfs.append(clf)
# # As we already know that there is a strong relation between, alpha and the depth of the tree.
# # We can find the relation using this plot.
# # print(f"Last node in Decision tree is {clfs[-1].tree_.node_count} and ccp_alpha for last node is {ccp_alphas[-1]}")
# # tree_depths = [clf.tree_.max_depth for clf in clfs]
# # plt.figure(figsize=(10,  6))
# # plt.plot(ccp_alphas[:-1], tree_depths[:-1])
# # plt.xlabel("effective alpha")
# # plt.ylabel("total depth")
# # plt.show()
#
#
#
# # train_scores = [clf.score(X_train, y_train) for clf in clfs]
# # test_scores = [clf.score(X_test, y_test) for clf in clfs]
# # fig, ax = plt.subplots()
# # ax.set_xlabel("alpha")
# # ax.set_ylabel("accuracy")
# # ax.set_title("Accuracy vs alpha for training and testing sets")
# # ax.plot(ccp_alphas, train_scores, marker='o', label="train",drawstyle="steps-post")
# # ax.plot(ccp_alphas, test_scores, marker='o', label="test",drawstyle="steps-post")
# # ax.legend()
# # plt.show()
#
#
#
# acc_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]
# #
# # tree_depths = [clf.tree_.max_depth for clf in clfs]
# plt.figure(figsize=(10,  6))
# plt.grid()
# plt.plot(ccp_alphas[:-1], acc_scores[:-1])
# plt.xlabel("effective alpha")
# plt.ylabel("Accuracy scores")
# plt.show()
# # We can clearly see that somewhere around 0.013 alpha, we get a
# # very good value of accuracy.
#
#
#
# # Prunned tree
#
# clf=DecisionTreeClassifier(random_state=0,ccp_alpha=0.009)
# clf.fit(X_train,y_train)
# plt.figure(figsize=(12,8))
# tree.plot_tree(clf,rounded=True,filled=True)
# plt.show()
#
# print(f'Accuracy of pruned tree {accuracy_score(y_test,clf.predict(X_test))}')
#
#
# #=============================
# # 2. Pre-Pruning :
# #===========================
# # This technique is used before construction of decision tree.
# # Pre-Pruning can be done using Hyperparameter tuning.
# # Overcome the overfitting issue.
# # In this blog i will use GridSearchCV for Hyperparameter tunin
# #
# # Lets’ take an example of Decision tree. When we build a DT model
# # we don’t have any idea about which criterion (“gini” or “entropy”)
# # ,what min_depth , what min_samples_split etc will give better model
# # so to break this kind of ambiguity we use hyperparameter tuning in
# # which we take a range of value for each parameters and whichever
# # parameteric value will be best we will feed that particular value
# # into DecisionTreeClassifier() .
#
# grid_param={"criterion":["gini","entropy"],
#              "splitter":["best","random"],
#              "max_depth":range(2,10,1),
#              "min_samples_leaf":range(1,5,1),
#              "min_samples_split":range(2,5,1)
#             }
# grid_search=GridSearchCV(estimator=clf,param_grid=grid_param,cv=5,n_jobs=-1)
# # estimator-> the classification model you have used,cv=5 -> we have divided our
# # dataset into five chunks,n_jobs=-1 ->we have taken default iteration
# grid_search.fit(X_train,y_train)
# print(grid_search.best_params_)
#
# # Pre-Pruning Operation :
# clf=DecisionTreeClassifier(criterion= 'entropy',max_depth= 9,min_samples_leaf= 1,min_samples_split= 3,splitter= 'random')
# clf.fit(X_train,y_train)
# y_predicted=clf.predict(X_test)
# print(f'pre tuned tree {accuracy_score(y_test,y_predicted)}')
# plt.figure(figsize=(20,12))
# tree.plot_tree(clf,rounded=True,filled=True)
# plt.show()