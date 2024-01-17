import pandas as pd
class predictSales:
    def loadcsv(self):
        return pd.read_csv("./data/train/sales.csv")
    def __init__(self):
        self.loadData=self.loadcsv()
        print(self.loadData)
obj=predictSales()
