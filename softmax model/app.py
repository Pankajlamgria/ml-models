import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class score():
    def __init__(self):
        self.loaddata()
        self.predictsclearn()
        self.predict()
        self.test()
    def predictsclearn(self):
        softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10, random_state=36)
        softmax_reg.fit(self.xtrain,self.ytrain)
        y_pred = softmax_reg.predict(self.xtest)
        zero,one,two=0,0,0
        for i in range(len(self.ytest)):
            if(self.ytest[i]==y_pred[i]):
                if(self.ytest[i]==0):
                    zero+=1
                elif(self.ytest[i]==1):
                    one+=1
                else:
                    two+=1
        print("zero:",zero,',','One:',one,"two:",two)
        accuracy = accuracy_score(self.ytest, y_pred)
        print("Accuracy(sclearn):",accuracy)
    def test(self):
        x0=np.ones((self.xtest.shape[0],1))
        self.xtest=np.hstack((x0,self.xtest))
        mult=self.xtest.dot(self.weight)
        exponents=np.exp(mult)
        exponents=np.nan_to_num(exponents,nan=1,posinf=99999)         
        prediction=exponents/exponents.sum(axis=1,keepdims=True)
        ypred=np.argmax(prediction,axis=1)
        print("Accuracy : ",accuracy_score(self.ytest,ypred))
    def visualize(self,y):
        arr=np.array(y.value_counts())
        l=[1,0,2]
        plt.pie(arr,labels=l,explode=[.1,0,0],autopct='%.2f%%')
        plt.show()
    def onehot(self):
        output=np.zeros((len(self.ytrain),self.container))
        for i in range(len(self.ytrain)):
            output[i][self.ytrain[i]]=1
        self.ytrain=output
    def predict(self):
        x0=np.ones((self.xtrain.shape[0],1))
        self.onehot()
        self.xtrain=np.hstack((x0,self.xtrain))
        weight=np.zeros((self.xtrain.shape[1],self.container))
        alpha,epoc=.001,1000
        for i in range(epoc):
            xtQ=self.xtrain.dot(weight)
            xtQ=np.exp(xtQ)
            xtQ=np.nan_to_num(xtQ,nan=0,posinf=999999)
            sumarr=xtQ.sum(axis=1)
            sumarr=sumarr.reshape(len(self.ytrain),1)
            prediction=xtQ/sumarr
            gradient=self.xtrain.T.dot(prediction-self.ytrain)
            weight-=alpha*gradient
        self.weight=weight

        print(weight)
        plt.plot(weight.T[0],label="Q0")
        plt.plot(weight.T[1],label="Q1")
        plt.plot(weight.T[2],label="Q2")
        plt.legend()
        plt.show()


    def loaddata(self):
        df=pd.read_csv("./credit_score.csv")
        y=df["Credit_Score"]
        x=df.drop(['Credit_Score',"Unnamed: 0","Month","Age"],axis=1)
        self.container=len(y.unique())
        # self.visualize(y)
        
        sc=StandardScaler()
        x=sc.fit_transform(x)
        self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(x,y,test_size=.2,random_state=42)
        self.xtrain=np.array(self.xtrain)
        self.ytrain=np.array(self.ytrain)
        self.xtest=np.array(self.xtest)
        self.ytest=np.array(self.ytest)

obj=score()