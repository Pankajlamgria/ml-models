import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
class svm():
    def __init__(self):
        self.loaddata()
        self.fit(.01,.01,1)
        self.sklearn()
    def sklearn(self):
        svm_model = SVC(kernel='linear')
        svm_model.fit(self.xtrain,self.ytrain)
        y_pred = svm_model.predict(self.xtest)
        accuracy = accuracy_score(self.ytest, y_pred)
        print(f"Accuracy: {accuracy*100:.2f}%")
    def predict(self,weight,b):
        fx=self.xtest.dot(weight)
        fx-=b
        ypred=np.array([])
        for i in range(len(fx)):
            if(fx[i]<=0):
                ypred=np.append(ypred,-1)
            else:
                ypred=np.append(ypred,1)
        first,second=0,0
        for i in range(len(ypred)):
            if(ypred[i]==self.ytest[i]):
                if(ypred[i]==1):
                    first+=1
                else:
                    second+=1
        print('Accuracy:',(first+second)*100/len(self.ytest),"%")
        
    def fit(self,alpha,lamda,epoc):
        feature=len(self.xtrain[0])
        weight=np.zeros((feature,1))
        b=0
        for i in range(epoc):
            fx=self.xtrain.dot(weight)
            fx-=b
            temp=self.ytrain
            temp=temp.reshape((len(self.ytrain),1))
            fx*=temp
            for j in range(len(self.ytrain)):
                if(fx[j][0]>=1):
                    weight-=alpha*2*lamda*(weight)
                else:
                    t1=(2*lamda*(weight))
                    t2=self.ytrain[j]*(self.xtrain[j].T)
                    t2=t2.reshape((feature,1))
                    weight-=alpha*(t1-t2)
                    b-=alpha*self.ytrain[j]
        self.predict(weight,b)
    def preprocess(self,df):
        df=df.drop(['Loan_ID'],axis=1)
        df['Gender']=df['Gender'].fillna('Male')
        df['Married']=df['Married'].fillna("Yes")
        df['Dependents']=df['Dependents'].fillna("0")
        df['Self_Employed']=df['Self_Employed'].fillna('No')
        df.dropna(inplace=True)
        df['Gender']=df['Gender'].replace(['Male','Female'],[1,0])
        df['Education']=df['Education'].replace(['Graduate','Not Graduate'],[1,0])
        df['Married']=df['Married'].replace(['Yes','No'],[1,0])
        df['Self_Employed']=df['Self_Employed'].replace(['Yes','No'],[1,0])
        df['Loan_Status']=df['Loan_Status'].replace(['Y','N'],[1,-1])
        df['Property_Area']=df['Property_Area'].replace(['Semiurban','Urban','Rural'],[3,2,1])
        df['Dependents']=df['Dependents'].replace('3+',"5")
        df['Dependents']=df['Dependents'].astype('float')
        return df
        
    def loaddata(self):
        df=pd.read_csv('./train.csv')
        df=self.preprocess(df)
        sc=StandardScaler()
        y=df['Loan_Status']
        x=df.drop(['Loan_Status'],axis=1)
        x=sc.fit_transform(x)
        x=pd.DataFrame(x)
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=.7,random_state=42)
        self.xtrain=np.array(xtrain)
        self.xtest=np.array(xtest)
        self.ytrain=np.array(ytrain)
        self.ytest=np.array(ytest)
obj=svm()