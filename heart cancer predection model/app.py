import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
# from matplotlib.pyplot import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class heart():
    def __init__(self):
        self.loaddata()
        self.Q=np.zeros(len(self.xtrain[0]))
        self.gradientdescent(.001)
        self.predict()
    def sigmoid(self,z):
        return 1/(1+math.exp(-z))
    def diff(self):
        gq=np.array([])
        for i in range(len(self.xtrain)):
            val=self.sigmoid(np.dot(self.Q,self.xtrain[i]))
            gq=np.append(gq,val)
        # print(gq)
        return gq
    def predict(self):
        count=0
        for i in range(len(self.ytest)):
            val=self.sigmoid(np.dot(self.Q,self.xtest[i]))
            val=round(val)
            count+=abs(val-self.ytest[i])
        print("Accuracy:",1-count/len(self.ytest))

    def gradientdescent(self,alpha):
        for i in range(100):
            h=self.diff()
            temp=h-self.ytrain
            newmat=np.dot(self.xtrain.T,temp)
            self.Q=self.Q-alpha*newmat
        print(self.Q)
    def preprocess(self,x):
        m=x['glucose'].mean()
        x['glucose'].fillna(value=m,inplace=True)
        x['heartRate'].fillna(value=x['heartRate'].mean(),inplace=True)
        x['BMI'].fillna(value=x['BMI'].mean(),inplace=True)
        x=x.dropna()
        return x
    def loaddata(self):
        df=pd.read_csv("framingham.csv")
        x=df.drop(["education"],axis=1)
        x=self.preprocess(x)
        y=x['TenYearCHD']
        x=x.drop(['TenYearCHD'],axis=1)
        one=np.ones(len(x))
        sc=StandardScaler()
        x=sc.fit_transform(x)
        x=pd.DataFrame(x)
        x.insert(0,'X0',one)
        self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(x,y,test_size=.2,random_state=10)
        self.xtrain=np.array(self.xtrain)
        self.xtest=np.array(self.xtest)
        self.ytrain=np.array(self.ytrain)
        self.ytest=np.array(self.ytest)


obj=heart()