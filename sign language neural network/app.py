import numpy as np
import pandas as pd

def loaddata():
    df=pd.read_csv('./train.csv')
    ytrain=df['label']
    xtrain=df.drop(["label"],axis=1)
    df=pd.read_csv('./test.csv')
    ytest=df['label']
    xtest=df.drop(['label'],axis=1)
    return np.array(xtrain),np.array(ytrain),np.array(xtest),np.array(ytest)

def var_creation(x):
    row,col=x.shape
    w1=np.random.randn(50,row)*np.sqrt(2/col)
    b1=np.random.randn(50,1)*np.sqrt(2/col)
    w2=np.random.randn(25,50)*np.sqrt(2/col)
    b2=np.random.randn(25,1)*np.sqrt(2/col)
    return w1,b1,w2,b2
def relu(z1):
    return np.maximum(z1,0)
def softmax(z2):
    return np.exp(z2)/sum(np.exp(z2))
def forward_propogation(x,w1,b1,w2,b2):
    z1=w1.dot(x)+b1
    a1=relu(z1)
    z2=w2.dot(a1)+b2
    a2=softmax(z2)
    return z1,a1,z2,a2
def onehot(y):
    output=np.zeros((25,len(y)))
    for i in range(len(y)):
        output[y[i]][i]=1
    return output
def deri_relu(z1):
    return z1>0
def backward_propogation(w2,y,a1,z1,a2,z2,x):
    onehot_y=onehot(y)
    sample=x.shape[1]
    dl2=a2-onehot_y
    dl1=(((w2.T).dot(dl2))*deri_relu(z1))
    dj_w2=(1/sample)*(dl2.dot(a1.T))
    dj_b2=(1/sample)*np.sum(dl2,axis=1).reshape(25,1)
    dj_w1=(1/sample)*dl1.dot(x.T)
    dj_b1=(1/sample)*np.sum(dl1,axis=1).reshape(50,1)
    return dj_w2,dj_b2,dj_w1,dj_b1

def update_params(w1,w2,b1,b2,alpha,dj_w2,dj_b2,dj_w1,dj_b1):
    w1=w1-alpha*(dj_w1)
    b1=b1-alpha*(dj_b1)
    w2=w2-alpha*(dj_w2)
    b2=b2-alpha*(dj_b2)
    return w1,b1,w2,b2

def calc_acc(pred,y):
    pred=pred.T
    pred_res=[np.argmax(x) for x in pred]
    print(np.sum(pred_res==y)/len(y))
    
def gradient(epoc,alpha,x,y):
    w1,b1,w2,b2=var_creation(x)
    for _ in range(epoc):
        z1,a1,z2,a2=forward_propogation(x,w1,b1,w2,b2)
        dj_w2,dj_b2,dj_w1,dj_b1=backward_propogation(w2,y,a1,z1,a2,z2,x)
        w1,b1,w2,b2=update_params(w1,w2,b1,b2,alpha,dj_w2,dj_b2,dj_w1,dj_b1)
        if(_%50==0):
            calc_acc(a2,y)
    return w1,b1,w2,b2

xtrain,ytrain,xtest,ytest=loaddata()
xtrain=xtrain.T
xtest=xtest.T
xtrain=xtrain/255
xtest=xtest/255
w1,b1,w2,b2=gradient(2000,.1,xtrain,ytrain)

z1,a1,z2,a2=forward_propogation(xtest,w1,b1,w2,b2)
calc_acc(a2,ytest)