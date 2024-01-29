import numpy as np

class node:
    def __init__(self,feature=None,threshold=None,right=None,left=None,depth=0,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.depth=depth
        self.right=right
        self.left=left
        self.value=value
    def is_leaf(self):
        return (self.value is not None)


class regression:
    def __init__(self,depth=10,max_sample=3):
        self.depth=depth
        self.max_no_sample=max_sample
    def fit(self,x,y):
        self.root=self.create_node(x,y,0)
    
    def create_node(self,x,y,depth):
        # base condition
        no_sample,feature=x.shape
        uniq=len(np.unique(y))
        
        if(depth>=self.depth or no_sample<=self.max_no_sample or uniq==1):
            avg=np.sum(y)/len(y)
            return node(value=avg)

        # finding best suitable feature and threshold
        best_feature,threshold=self.best_split(x,y,feature)

        left_indx,right_indx=self.split(x[:,best_feature],threshold)
        #create left and right child
        left=self.create_node(x[left_indx,:],y[left_indx],depth+1)
        right=self.create_node(x[right_indx,:],y[right_indx],depth+1)
        return node(best_feature,threshold,right,left,depth)

    
    def best_split(self,x,y,feats):
        mse,feature,threshold=np.Inf,None,None
        for i in range(feats):
        
            error,th=self.min_error(x[:,i],y)
            if(error<mse):
                mse=error
                feature=i
                threshold=th
        return feature,threshold

    def sort(self,col,y):
        sample=len(y)
        indx=np.argsort(col)
        newcol=np.zeros(sample)
        newy=np.zeros(sample)
        for i in range(sample):
            newcol[i]=col[indx[i]]
            newy[i]=y[indx[i]]
        return newcol,newy
    
    def min_error(self,col,y):
        newcol,newy=self.sort(col,y)
        sample=len(y)
        mse_E,threshold=np.Inf,None
        for i in range(sample-1):
            avg=(newcol[i]+newcol[i+1])/2
            left_indx,right_indx=self.split(newcol,avg)
            mse_error=self.mean_square_error(newy,left_indx,right_indx)
            if(mse_error<mse_E):
                mse_E=mse_error
                threshold=avg
        return mse_E,threshold

    def split(self,newcol,avg):
        left_indx=np.argwhere(newcol<=avg).flatten()
        right_indx=np.argwhere(newcol>avg).flatten()
        return left_indx,right_indx
    
    def mean_square_error(self,y,left_indx,right_indx):
        
        left_arr=y[left_indx]
        right_arr=y[right_indx]
        left_avg,right_avg=np.sum(left_arr)/len(left_indx),np.sum(right_arr)/len(right_indx)
        square_error=np.sum((left_arr-left_avg)**2)
        square_error+=np.sum((right_arr-right_avg)**2)
        return square_error/len(y)
    def predict(self,x):
        ans=np.array([self.traverse(i,self.root) for i in x])
        return ans
    def traverse(self,x,root):
        if(root.is_leaf()):
            return root.value
        if(x[root.feature]<=root.threshold):
            return self.traverse(x,root.left)
        return self.traverse(x,root.right)
        
        

            
