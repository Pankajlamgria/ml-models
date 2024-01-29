import numpy as np
import statistics as st
class node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,res=None):
        self.feature=feature
        self.threshold=threshold
        self.right=right
        self.left=left
        self.res=res
    def is_leaf(self):
        if(self.res==None):
            return 0
        return 1

class decision():
    def __init__(self,depth=10,min_sample=2,random_feature=None):
        self.depth=depth
        self.min_sample=min_sample
        self.random_feature=random_feature
    def fit(self,x,y):
        if(not self.random_feature or self.random_feature>x.shape[1]):
            self.random_feature=x.shape[1]
        self.root=self.create_node(x,y)
    def create_node(self,x,y,depth=0):
        no_of_sample,feature=x.shape
        n_labels=len(np.unique(y))
        if(depth>=self.depth or n_labels==1 or no_of_sample<=self.min_sample):
            output=st.mode(y)
            return node(res=output)
        
        # find best feature,threshold
        random_feats=np.random.choice(feature,self.random_feature,replace=False)
        best_feature,threshold=self.best_split(x,y,random_feats)
        
        left_indx,right_indx=self.split(x[:,best_feature],threshold)
        
        # creating child
        left=self.create_node(x[left_indx,:],y[left_indx],depth+1)
        right=self.create_node(x[right_indx,:],y[right_indx],depth+1)
        return node(best_feature,threshold,left,right)
        
        
    def best_split(self,x,y,feats):
        best_information=-1
        best_feature=None
        threshold=None
        for i in feats:
            poss_treshold=np.unique(x[:,i])
            for th in poss_treshold:
                If_gain=self.info_gain(x[:,i],y,th)
                if(If_gain>best_information):
                    best_information=If_gain
                    best_feature=i
                    threshold=th
        return best_feature,threshold
    
    def info_gain(self,col,y,threshold):
        len_y=len(y)
        parent=self.entropy(y)
        left,right=self.split(col,threshold)
        left_entropy=self.entropy(y[left])
        right_entropy=self.entropy(y[right])
        
        child_entropy=((len(left)/len_y)*left_entropy+((len(right)/len_y)*right_entropy))
        return parent-child_entropy
    
    def split(self,col,threshold):
        left_indx=np.argwhere(col<=threshold).flatten()
        right_indx=np.argwhere(col>threshold).flatten()
        return left_indx,right_indx
    
    def entropy(self,y):
        total=len(y)
        arr=np.bincount(y)
        arr=arr/total
        return -np.sum([x*np.log(x) for x in arr if(x>0)])

    def predict(self,xtest):
        return np.array([self.traverse(x,self.root) for x in xtest])
    
    def traverse(self,x,root):
        if(root.is_leaf()):
            return root.res
        if(x[root.feature]<=root.threshold):
            return self.traverse(x,root.left)
        return self.traverse(x,root.right)

