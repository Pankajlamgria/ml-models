import sklearn.datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from regressionTree import regression
import matplotlib.pyplot as plt

def loaddata():
    df=pd.read_csv("./advertisingsales.csv")
    y=df['Sales ($)']
    x=df.drop(['Unnamed: 0','Sales ($)'],axis=1)
    sc=StandardScaler()
    x=sc.fit_transform(x)
    x=pd.DataFrame(x)
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=.8,random_state=42)
    xtrain=np.array(xtrain)
    xtest=np.array(xtest)
    ytrain=np.array(ytrain)
    ytest=np.array(ytest)
    return xtrain,xtest,ytrain,ytest

xtrain,xtest,ytrain,ytest=loaddata()
sale=regression(depth=20,max_sample=2)
sale.fit(xtrain,ytrain)
ypred=sale.predict(xtest)
print(ypred,ytest)

plt.plot(ypred,label="Predicted value")
plt.plot(ytest,label="True value")
plt.legend()
plt.show()