from sklearn.model_selection import train_test_split
from randomforest import Random_forest
import numpy as np
import pandas as pd

def loaddata():
    df=pd.read_csv("./test.csv")
    y=df['label']
    X=df.drop(['label'],axis=1)
    X=X/255
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    y_test=np.array(y_test)
    return X_train,X_test,y_train,y_test

def classify(X_train,X_test,y_train,y_test):
    clf = Random_forest(depth=5,tree=20,random_feature=150)
    clf.fit(X_train, y_train)
    predictions = clf._predict_(X_test)
    print("Training completed")
    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)
    acc = accuracy(y_test, predictions)
    print("Accuracy:",acc*100)

X_train,X_test,y_train,y_test=loaddata()
classify(X_train,X_test,y_train,y_test)
