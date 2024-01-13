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
        self.prediction(0.001,100)
        # self.predict()
    def preprocess(self,x):
        m=x['glucose'].mean()
        x['glucose'].fillna(value=m,inplace=True)
        x['heartRate'].fillna(value=x['heartRate'].mean(),inplace=True)
        x['BMI'].fillna(value=x['BMI'].mean(),inplace=True)
        x=x.dropna()
        return x
    def resdiff(self):
        hq=np.array([])
        for i in range(len(self.xtrain)):
            sum=self.Q[0]
            for j in range(len(self.xtrain.iloc[0,:])):
                sum+=self.Q[j+1]*self.xtrain.iloc[i,j]
            gz=1/(1+math.exp(-sum))
            gz=self.ytrain.iloc[i,]-gz
            hq=np.append(hq,gz)
        # print(hq.sum())
        return hq


    def sumof(self,diff,j):
        sum=0
        for i in range(len(self.xtrain)):
            sum+=diff[i]*self.xtrain.iloc[i,j]
        return sum
    def prediction(self,alpha,t):
        col=len(self.xtrain.iloc[0,:])
        self.Q=np.zeros(col+1)
        for i in range(t):
            diff=self.resdiff()
            self.Q[0]=self.Q[0]+alpha*(np.sum(diff))
            for j in range(col):
                self.Q[j+1]=self.Q[j+1]+alpha*(self.sumof(diff,j))
            print(self.Q)
        print(self.Q)
    def calclres(self):
        # out=np.array([])
        avg=0
        self.ytest=self.ytest.tail(10)
        for i in range(len(self.ytest)):
            out=np.array([])
            sum=self.Q[0]
            for j in range(len(self.xtest.iloc[0,:])):
                sum+=self.Q[j+1]*self.xtest.iloc[i,j]
            sum=1/(1+math.exp(-sum))
            print((sum))
            # out=np.append(out,round(sum))
            # avg+=(sum-self.ytest.iloc[i,])
        # print(avg)
        # print(self.ytest)
        # print(1-avg/len(self.ytest))
        # print(out)
        print(self.ytest)
    def loaddata(self):
        df=pd.read_csv("framingham.csv")
        x=df.drop(["education"],axis=1)
        x=self.preprocess(x)
        y=x['TenYearCHD']
        x=x.drop(['TenYearCHD'],axis=1)
        sc=StandardScaler()
        x=sc.fit_transform(x)
        x=pd.DataFrame(x)
        self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(x,y,test_size=.2,random_state=10)
    def predict(self):
        model=LogisticRegression()
        model.fit(self.xtrain,self.ytrain)
        ypred=model.predict(self.xtest)
        print(ypred)
        acc=accuracy_score(self.ytest,ypred)
        print(acc)
        



obj=heart()