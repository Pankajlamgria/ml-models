import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class spam():
    def __init__(self):
        self.loaddata()
        self.naive()
    def calcphi(self):
        return self.ytrain.sum()/len(self.ytrain)    
    def calcone(self):
        arr=np.array([])
        temp=self.xtrain.T
        for i in range(len(temp)):
            mult=temp[i].dot(self.ytrain).sum()
            arr=np.append(arr,mult/self.ytrain.sum())
        return arr            
    def calczero(self):
        temp=pd.DataFrame(self.ytrain)
        temp=temp.replace(1,2)
        temp=temp.replace(0,1)
        temp=temp.replace(2,0)   
        temp=np.array(temp)
        xtemp=self.xtrain.T
        arr=np.array([])
        for i in range(len(xtemp)):
            mult=(xtemp[i].dot(temp)).sum()
            arr=np.append(arr,mult/temp.sum())
        return arr
        
    def calcSuccess(self,phij1,phi):
        arr=np.array([])
        for i in range(len(self.xtest)):
            temp=self.xtest[i]*phij1
            temp[temp==0]=1.0
            newtemp=np.prod(temp)
            newtemp*=phi
            arr=np.append(arr,newtemp)
        return arr
                
    def accuracy(self,output):
        arr=np.array([])
        arr=abs(output-self.ytest).sum()
        print("Accuracy :",(1-arr/len(self.ytest))*100,'%')
        
    def predict(self,phi,phij1,phij0):
        pOfsuccess=self.calcSuccess(phij1,phi)
        pOfLost=self.calcSuccess(phij0,1-phi)
        output=np.array([])
        for i in range(len(pOfLost)):
            if(pOfLost[i]>pOfsuccess[i]):
                output=np.append(output,0)
            else:
                output=np.append(output,1)
        self.accuracy(output)                
                
    def naive(self):
        phi=self.calcphi()
        phijone=self.calcone()
        phijzero=self.calczero()
        self.predict(phi,phijone,phijzero)
    def loaddata(self):
        df=pd.read_csv("./emails.csv")
        y=df["Prediction"]
        x=df.drop(["Prediction","Email No."],axis=1) 
        sc=StandardScaler()
        sc.fit_transform(x)
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=.90,random_state=36)
        self.xtrain=np.array(xtrain)
        self.xtest=np.array(xtest)
        self.ytest=np.array(ytest)
        self.ytrain=np.array(ytrain)
        
    
        
        
        
obj=spam()