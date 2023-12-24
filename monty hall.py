import numpy as np
import random as rd
class montyhall:
    def __init__(self):
        print('''Welcome to monty Hall game:\nrule of game-Behind one door there is a Car and behind other TWO door there are goat\nyou will get whatever is present behind the door.''')
        self.arr=np.array(['Car',"Goat","Goat"])
        rd.shuffle(self.arr)
        self.start()
    def change(self,req,i):
        if(req==0 and i==1):
            return 2
        elif(req==0 and i==2):
            return 1
        elif(req==1 and i==0):
            return 2
        elif(req==1 and i==2):
            return 1
        elif(req==2 and i==0):
            return 1
        elif(req==2 and i==1):
            return 2
    def start(self):
        req=int(input("Enter the door to open:"))
        req-=1
        for i in range(len(self.arr)):
            if(i!=req and self.arr[i]=='Goat'):
                print("ThankGod you choosed ",req+1,"door because goat is in ",i+1,"door.")
                anotherchance=(input("\nYou have another chance weather you want to switch or stay with the door you choose(y/n):"))
                if(anotherchance=="y"):
                    req=self.change(req,i)
                    print("\nThanks changing.")
                else:
                    print("\nAlright")
                break
        if(self.arr[req]=="Car"):
            print("\nCongratulation,You won a Car.")
        else:
            print("\nYou won goat,Better Luck next time.")
obj=montyhall()