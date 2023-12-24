import numpy as np
import random as rd
from matplotlib import pyplot as plt
class montyhall:
    def __init__(self):
        self.arr=np.array(['Car',"Goat","Goat"])
        # rd.shuffle(self.arr)
        self.win=0
        self.lose=0
        self.winbyswitch=0
        self.windbystay=0
        self.req=np.array([])
        self.switchreq=np.array([])
        for i in range(10000):
            self.req=np.append(self.req,rd.randint(1,3))
            self.switchreq=np.append(self.switchreq,rd.randint(0,1))
        self.start()
    def change(self,req,i):
        if(req==0 and i==1):
            return 2
        elif(req==0 and i==2):
            return 1
        elif(req==1 and i==0):
            return 2
        elif(req==1 and i==2):
            return 0
        elif(req==2 and i==0):
            return 1
        elif(req==2 and i==1):
            return 0
    def start(self):
        for ind in range(len(self.req)):
            req=int(self.req[ind])
            req-=1
            anotherchance=self.switchreq[ind]
            for i in range(len(self.arr)):
                if(i!=req and self.arr[i]=='Goat'):
                    if(anotherchance==1):
                        req=self.change(req,i)
                    break
            if(self.arr[req]=="Car"):
                self.win+=1
                if(anotherchance==1):
                    self.winbyswitch+=1
                else:
                    self.windbystay+=1
            else:
                self.lose+=1
    def show(self):
        temparr=np.array([self.lose,self.win,self.winbyswitch,self.windbystay])
        temp2arr=np.array([len(self.req)/2,len(self.req)/2,(2/3)*len(self.req)/2,(1/3)*len(self.req)/2])
        plt.plot(temparr)
        plt.plot(temp2arr)
        plt.show()
    
        
obj=montyhall()
print(f"Total win:{obj.win}\nTotal lose:{obj.lose}\nWin by switch:{obj.winbyswitch}\nWin by stay:{obj.windbystay}")
obj.show()

