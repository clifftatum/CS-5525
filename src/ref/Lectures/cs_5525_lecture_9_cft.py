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
    eda = EDA('Clifford Tatum exploratoryDataAnalysis 1 Layer Neural Network (Essentially Logistic Regression) ')
    url = 'https://raw.githubusercontent.com/clifftatum/CS-5525-Term-Project/main/DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv'
    df = pd.read_csv(url)
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    features = cancer.feature_names
    target = cancer.target
    df = pd.DataFrame(cancer.data,columns=features)
    y =pd.Series(cancer.target)
    X = df
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    log_reg_model = LogisticRegression(max_iter=2500,random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=2,
                                                        shuffle=True)
    model = log_reg_model.fit(X,y)
    y_pred = model.predict(x_test)
    precision,recall, fscore, _ = precision_recall_fscore_support(y_test,y_pred,average='binary')

    print(f'{precision}')
    print(f'{recall}')
    print(f'{fscore}')

    cf_matrix = confusion_matrix(y_test,y_pred)
    sns.heatmap(cf_matrix,annot=True,cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # Confusion matrix

    from sklearn.linear_model import LinearRegression

    from tensorflow.keras.layers import InputLayer
    from keras.layers import Activation, Dense
    import tensorflow as tf
    import keras
    np.random.seed(42)
    tf.random.set_seed(42)

    ANN_model = keras.Sequential()
    ANN_model.add(InputLayer(input_shape = (30,)))
    ANN_model.add(Dense(1,activation = 'sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    ANN_model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    history = ANN_model.fit(x_train,y_train,epochs=10,
                            shuffle = False,
                            batch_size=32,
                            validation_split=0.2)
    fig = plt.figure()
    # Plot perf and accuracy
    plt.plot(history.history['accuracy'],label='Train')
    plt.plot(history.history['val_accuracy'],label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.show()

    fig = plt.figure()
    plt.plot(history.history['loss'],label='Train')
    plt.plot(history.history['val_loss'],label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.show()



    test_loss,test_acc = ANN_model.evaluate(x_test,y_test)
    print(f'{test_loss*100}')
    print(f'{test_acc}')








    pass
