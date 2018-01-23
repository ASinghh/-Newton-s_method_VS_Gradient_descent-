##In this file I define all my function to work in my IPYNB notebook
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

## define a function to create random dataset
def create_dataset_logistic_bionomial(dim,size):
    X = np.random.normal(loc=0.0, scale=10.0, size= (size,dim))
    Y = np.random.choice([0, 1], size=(size,), p=[1./2, 1./2])
    B = np.random.uniform(size=(dim,))
    return Y,X,B

##  define batch formation
def feed(X,Y,batch_size):
    assert len(X)%batch_size == 0 ##to make sure perfect allocation of data
    global cursor
    x_train = X[cursor:cursor+batch_size]
    y_train = Y[cursor:cursor+batch_size]
    if cursor == len(X)  :
        cursor = 0
    else :
        cursor += batch_size
    return x_train, y_train

#define activation function
def penalty_diff_sum(p):
    diff_list = []
    pen_list = []
    for i in p :
        pen_list.append(abs(i))
        if i == 0:
            diff_list.append(0)
        else:
            diff_list.append(int(i/abs(i)))
    return np.array(pen_list), np.array(diff_list)

#def hessian(w,x,b):
    
  #  w_v = w * ((1-w)+0.00001*np.random.rand(len(w), 11))
  #  return np.dot(x.T, x*w_v)
def hessian(beta, X):
    """
    Compute the Hessian X^TWX
    """
    w = 1.0 / (1 + np.exp((np.dot(X, beta))))
    w_vector = w * (1-w)
    
    return np.dot(X.T, X*w_vector)

def logits(B,x):
    logits = x.dot(B)
    return logits


def activation_sigmoid(p):
    return 1.0 / (1 + np.exp(-p))

def neg_log_likely(y,w):## finction borrowed from Aaron Webb, Gracias!
     l = -(np.nan_to_num(np.dot(y.transpose(), np.log(w))) + np.nan_to_num(np.dot((1-y).transpose(), np.log(1-w))))
     #l = -(np.dot(y.T, np.log(w)) + np.dot((1 - y).T, np.log(1-w)))  
     return l[0][0]

def neg_log_likely_regu(y,w,P,lamb):## finction borrowed from Aaron Webb, Gracias!
     
     l = -(np.nan_to_num(np.dot(y.transpose(), np.log(w)))+np.nan_to_num(np.dot((1-y).transpose(), np.log(1-w)))) + lamb*np.sum(P)
     return l
 
def gradient_regu(x,y,w,G,lamb) :
    gradient = ((x.T.dot(w-y))/len(x) + lamb*G)    ### why does it perform better?
    return gradient

def predict_binary(B,X,thresshold):
    logit = 1.0 / (1 + np.exp(-(X.dot(B))))
    for i in range(len(logit)):
        if logit[i] < thresshold:
            logit[i] = int(0)
        else :
            logit[i] = int(1)
    return logit.astype(int)


    
def accuracy(y_true,y_pred, j =0):
    assert len(y_true) == len(y_pred)
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            j += 1
    return j/len(y_true)
    

def gradient(x,y,w) :
    gradient = (x.T.dot(w-y))/len(x)   ### why does it perform better?
    return gradient

def SGD(X,Y,batch_size,step_size,lamb):
    x,y = feed(X,Y,batch_size)
    global B
    w =  activation_sigmoid(logits(B,x)) 
    P,G   = penalty_diff_sum(B)
    loss = neg_log_likely(y,w,P,lamb)                                         
    B += -(step_size *gradient(x,y,w,G,lamb))
    return loss
    


 


    
    
