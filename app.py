import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class score():
    def __init__(self):
        self.loaddata()
        self.predict()
    def predict(self):
        softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10, random_state=42)
        softmax_reg.fit(self.xtrain,self.ytrain)
        y_pred = softmax_reg.predict(self.xtest)
        accuracy = accuracy_score(self.ytest, y_pred)
        print(accuracy)
    def loaddata(self):
        df=pd.read_csv("./Score.csv")
        df=df.replace("Low_spent_Small_value_payments",11)
        df=df.replace("High_spent_Medium_value_payments",32)
        df=df.replace("High_spent_Large_value_payments",33)
        df=df.replace("Low_spent_Medium_value_payments",12)
        df=df.replace("High_spent_Small_value_payments",31)
        df=df.replace("Low_spent_Large_value_payments",13)
        df['Credit_Score']=df["Credit_Score"].replace("Standard",2)
        df['Credit_Score']=df["Credit_Score"].replace("Poor",1)
        df['Credit_Score']=df["Credit_Score"].replace("Good",3)
        df=df.replace("Yes",1)
        df=df.replace("NM",1)
        df=df.replace("No",0)
        df=df.drop(['Credit_Mix'],axis=1)
        y=df['Credit_Score']
        x=df.drop(['Credit_Score'],axis=1)
        sc=StandardScaler()
        x=sc.fit_transform(x)
        print(x)
        self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(x,y,test_size=.2,random_state=10)

obj=score()