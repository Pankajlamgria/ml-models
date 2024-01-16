import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
class cancer():
    def __init__(self):
        self.loaddata()
        self.gda()
    def calcphi(self):
        sum=0
        for i in self.ytrain:
            if(i==1):
                sum+=1
        return sum/len(self.ytrain)
    def calcu1(self,n):
        res=np.zeros(len(self.xtrain[0]))
        for i in range(len(self.ytrain)):
            if(self.ytrain[i]==1):
                res+=self.xtrain[i] 
        return res/n
    def calcu0(self,n):
        res=np.zeros(len(self.xtrain[0]))
        for i in range(len(self.ytrain)):
            if(self.ytrain[i]==0):
                res+=self.xtrain[i]
        return res/n
            

    def calcsigma(self,u1,u0):
        val=np.zeros((9,9))
        for i in range(len(self.ytrain)):
            if(self.ytrain[i]==1):
                val+=(self.xtrain[i].reshape(9,1)-u1).dot((self.xtrain[0].reshape(9,1)-u1).T)
            else:
                val+=(self.xtrain[i].reshape(9,1)-u0).dot((self.xtrain[0].reshape(9,1)-u0).T)
        return val/len(self.ytrain)
    def prob(self,sigma,u,x):
        temp=(x.reshape(9,1)-u)
        temp1=np.linalg.inv(sigma).dot(temp)
        val=(temp.T).dot(temp1)
        det=(np.linalg.det(sigma))
        val=round(float(val))
        return np.exp(val/(-2))/((2*math.pi)**(9/2)*(det**(1/2)))
    
    def output(self,sigma,u0,u1,phi,x):
        pofs=self.prob(sigma,u1,x)*phi
        pofl=self.prob(sigma,u0,x)*(1-phi)
        if(pofs>pofl):
            return 1
        return 0
    def predict(self,sigma,u1,u0,phi):
        ans=np.array([])
        for i in range(len(self.ytest)):
            ans=np.append(ans,self.output(sigma,u0,u1,phi,self.xtest[i]))
        print("\nAccuracy:",1-abs(ans-self.ytest).sum()/len(self.ytest))
    def gda(self):
        success_count=self.ytrain.sum()
        phi=self.calcphi()
        u1=self.calcu1(success_count).reshape(len(self.xtrain[0]),1)
        u0=self.calcu0(len(self.ytrain)-success_count).reshape(len(self.xtrain[0]),1)
        sigma=self.calcsigma(u1,u0)
        self.predict(sigma,u1,u0,phi)


    def loaddata(self):
        df=pd.read_csv("./data.csv")
        y=df['Class']
        y=y.replace(2,0)
        y=y.replace(4,1)
        x=df.drop(['Sample code number',"Class"],axis=1)
        sc=StandardScaler()
        x=pd.DataFrame(sc.fit_transform(x)) 
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=.8,random_state=20)
        self.xtrain=np.array(xtrain)
        self.xtest=np.array(xtest)
        self.ytrain=np.array(ytrain)
        self.ytest=np.array(ytest)
obj=cancer()