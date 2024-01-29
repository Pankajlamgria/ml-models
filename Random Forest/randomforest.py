from Decision import decision
import numpy as np
import statistics as st
class Random_forest:
    def __init__(self,depth=10,tree=10,min_sample=2,random_feature=None):
        self.depth=depth
        self.no_tree=tree
        self.min_sample=min_sample
        self.random_feature=random_feature
        self.tree=[]
    def fit(self,x,y):
        for _ in range(self.no_tree):
            x,y=self.bootstrap(x,y)
            clf=decision(depth=self.depth,min_sample=self.min_sample,random_feature=self.random_feature)
            clf.fit(x,y)
            self.tree.append(clf)

    def bootstrap(self,x,y):
        n=x.shape[0]
        indx=np.random.choice(n,n,replace=True)
        x=x[indx]
        y=y[indx]
        return x,y
    
    def _predict_(self,x):
        tree_res=[t.predict(x) for t in self.tree]
        tree_res=np.array(tree_res)
        tree_res=tree_res.T
        ans=[st.mode(ypred) for ypred in tree_res]
        return ans
