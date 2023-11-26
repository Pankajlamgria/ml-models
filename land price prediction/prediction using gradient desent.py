import numpy as np
import matplotlib.pyplot as plt

class landprice:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.q=[0,0,0]
        
    def predict(self):
        fx=np.array([])
        for diffdata in self.x:
            fx=np.append(fx,self.q[0]+self.q[1]*diffdata[1]+self.q[2]*diffdata[2])
        return fx
    
    def calculateJq(self):  
        val=self.predict()
        print("sum:",1/2*(np.sum(val-self.y)**2))
        return 1/2*(np.sum(val-self.y)**2)
    
    def calchx(self,x):
        print("functions value:",self.q[0]+self.q[1]*x[1]+self.q[2]*x[2])
        return self.q[0]+self.q[1]*x[1],self.q[2]*x[2]
    
    def updatevalues(self,rate):
        val=self.predict()
        
        self.q[0]=self.q[0]-rate*(np.sum(val-self.y)*len(self.y))
        self.q[1]=self.q[1]-rate*(np.sum(val-self.y)*(self.x[0][1]+self.x[1][1]+self.x[2][1]+self.x[3][1]+self.x[4][1]))
        self.q[2]=self.q[2]-rate*(np.sum(val-self.y)*(self.x[0][2]+self.x[1][2]+self.x[2][2]+self.x[3][2]+self.x[4][2]))
    def plot_best_fit(self, value,testing, fig):
        f = plt.figure(fig)
        plt.plot(testing,self.y, color='b')
        plt.plot(testing, value, color='g')
        f.show()
        plt.show()
rate=0.0000001
y=np.array([5003,7503,5009,10001,6007])
x=np.array([1,100,4,1,150,6,1,100,1,1,200,2,1,120,1])
x=np.resize(x,(5,3))
obj=landprice(x,y)
print(obj.predict())
for i in range(100):
    obj.updatevalues(rate)
print(obj.y)
print(obj.predict())
obj.plot_best_fit(obj.predict(),np.array([1,2,3,4,5]),"this is the result")

