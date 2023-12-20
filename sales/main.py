import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
class sales():
    def __init__(self):
        self.loaddata()
        self.scale()
        self.Q=np.array([0,0,0,0],dtype=float)
        self.gradient(0.001,100)
        self.res()
    def costfunction(self):
        val=self.calculateTheta()
        print("Cost function result : J(Q)=",float((np.square(val).sum())/2))
    def res(self):
        x=np.array((),dtype=float)
        for i in range(len(self.Xtest)):
            val=self.Q[0]
            for j in range(1,len(self.Q)):
                val+=self.Q[j]*self.Xtest[i][j-1]
            x=np.append(x,val)
        y=np.array(self.ytest)
        out=np.sum(x-y)
        print("difference avg:",abs(out.sum())/len(self.Xtest))
        self.costfunction()
        plt.plot(x,label="line 1")
        plt.plot(y,label="line 2")
        plt.legend()
        plt.show()
            
    def calculateTheta(self):
        temptheta=np.array([],dtype=float)
        for i in range(len(self.Xtrain)):
            val=self.Q[0]
            for j in range(1,len(self.Q)):
                val+=self.Q[j]*self.Xtrain[i][j-1]
            temptheta=np.append(temptheta,val-self.ytrain.iloc[i])
        return temptheta
    def multhq(self,theta,j):
        s=0.0
        for i in range(len(theta)):
            s+=theta[i]*self.Xtrain[i][j]
        return s
    def gradient(self,rate,count):
        for time in range(count):
            theta=self.calculateTheta()
            for j in range(0,len(self.Q)):
                if(j==0):
                    self.Q[j]=self.Q[j]-rate*(theta.sum())
                else:
                    val=self.Q[j]-rate*self.multhq(theta,j-1)
                    self.Q[j]=val
    def split(self,X,y):
        self.Xtrain,self.Xtest,self.ytrain,self.ytest=train_test_split(X,y,train_size=.8,random_state=40)
        
    def loaddata(self):
        df=pd.read_csv("./advertisingsales.csv")
        df.drop(["Unnamed: 0"],axis=1,inplace=True)
        y=df["Sales ($)"]
        X=df.drop(["Sales ($)"],axis=1)
        self.split(X,y)
    
    def scale(self):
        sc=StandardScaler()
        self.Xtrain=sc.fit_transform(self.Xtrain)
        self.Xtest=sc.fit_transform(self.Xtest)
obj=sales()

