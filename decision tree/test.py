from sklearn import datasets
from sklearn.model_selection import train_test_split
from Decision import decision
import numpy as np
import pandas as pd
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
clf = decision(depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Training completed")
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)