import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import tree

np.random.seed(123)
df = sns.load_dataset('titanic')
df.dropna(how='any', inplace=True)
df = df.drop(columns=['sex', 'embarked', 'class',
                      'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'])

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['survived']),
                                                              df['survived'],
                                                              test_size=.2,
                                                              random_state=123)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=123)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
X_train = X_train.select_dtypes(include=numerics)
X_test = X_test.select_dtypes(include=numerics)

clf.fit(X_train, y_train)
# depth of the decision tree
print('Depth of the Decision Tree :', clf.get_depth())

# predict the target on the train dataset
predict_train = clf.predict(X_train)
print('Target on train data', predict_train)

# Accuray Score on train dataset
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train, predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = clf.predict(X_test)
print('Target on test data', predict_test)
# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_test, predict_test)
print('accuracy_score on test dataset : ', accuracy_test)


tuned_parameters = [{'max_depth': [1, 2, 3, 4, 5],
                     'min_samples_split': [2, 4, 6, 8, 10],
                     'min_samples_leaf': [i for i in range(1, 50)],
                     'max_features': [4],
                     'splitter': ['best', 'random'],
                     'criterion': ['gini', 'entropy', 'log_loss']}]

scores = ['recall']

for score in scores:

    print()
    print(f"Tuning hyperparameters for {score}")
    print()

    clf = GridSearchCV(
        DecisionTreeClassifier(), tuned_parameters,
        scoring=f'{score}_macro'
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds,
                                 clf.cv_results_['params']):
        print(f"{mean:0.3f} (+/-{std * 2:0.03f}) for {params}")


# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# cancer = load_breast_cancer()
# df = pd.DataFrame(cancer.data,
#                   columns=cancer.feature_names)
#
# features = cancer.feature_names
# y = pd.Series(cancer.target)
# X = df
# print(df.isnull().sum().sum())
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
#                                                     shuffle=True, random_state=2)
# log_reg_model = LogisticRegression(max_iter=2500,
#                                    random_state=42)
# log_reg_model.fit(X_train, y_train)
#
# # Make predictions
# y_pred = log_reg_model.predict(X_test)
#
# # Model evaluation
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_recall_fscore_support
# import numpy as np
# print(f"Accuracy:{accuracy_score(y_test, y_pred):.2f}")
# precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred,
#                                                                average='binary')
#
# print(f"Precision: {precision:.2f}")
# print(f"Recall:{recall:.2f}")
# print(f"F-Score:{fscore:.2f}")
# #====================
# # Confusion Matrix
# #====================
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
# cf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(cf_matrix, annot=True, cmap='Blues')
# plt.xlabel('Predicted', fontsize=12)
# plt.ylabel('True', fontsize=12)
# plt.show()
# #======================================================================================
# # Build the same logistic regression model with a neural network mindset in Keras
# #=======================================================================================
# from keras.models import Sequential
# from tensorflow.keras.layers import InputLayer
# from tensorflow.keras.layers import Dense
# import tensorflow as tf
# np.random.seed(42)
# tf.random.set_seed(42)
#
# # Step 1: Define the neural network architecture
# # Step 2: Instantiate a model of the Keras Sequential() class
# ANN_model = Sequential()
# ANN_model.add(InputLayer(input_shape=(30, )))
# # No hidden layers
# ANN_model.add(Dense(1, activation='sigmoid'))
#
# #===========================
# # Step 5: Compile the model
# #============================
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
# ANN_model.compile(optimizer=optimizer,
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
# # Step 6: Fit the model
# history = ANN_model.fit(X_train, y_train,
#                         epochs=1000, batch_size=32,
#                         validation_split=0.2,
#                         shuffle=False)
# # Step 7: Plot the performance of the model during training
# plt.plot(history.history['accuracy'], label='Train')
# plt.plot(history.history['val_accuracy'], label='Validation')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.title('Model Accuracy')
# plt.legend(loc='upper left')
# plt.show()
#
# plt.plot(history.history['loss'], label='Train')
# plt.plot(history.history['val_loss'], label='Validation')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.title('Model Loss')
# plt.legend(loc='upper right')
# plt.show()
#
# # Step 8: Evaluate the model on the test data
# test_loss, test_acc = ANN_model.evaluate(X_test, y_test)
# print(f"Test loss:{test_loss*100:.2f}")
# print(f"Test accuracy:{test_acc:.2f}%")