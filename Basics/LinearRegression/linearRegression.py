#Creating a linear equatation that best fits that given data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def PlotData(a,b):
    #print(X.shape)
    #print(t.shape)
    plt.figure(figsize=(10, 5))
    plt.scatter(X, t)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

def PlotDataWithEQ(x,y,m,b):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y)
    x_axis = np.linspace(x.min(), x.max(), N)
    y_axis = x_axis*m + b
    plt.plot(x_axis, y_axis)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

#least mean squares
def LMSoffline(X,t,N):
    # sum of : X, t ,X*t ,X^2
    # y = mx + b
    # slope : m = (nsumof(xy) - sumof(x) * sumof(y))/(n(sumof(x^2) - (sumof(x))^2))
    # when x = 0, y : b = (sumof(y) - m - sumof(x))/n
    array  = np.zeros(4)
    
    print(array)
    for i in range(N):
        array[0] = array[0] + X[i]*t[i]
        array[1] = array[1] + X[i]
        array[2] = array[2] + t[i]
        array[3] = array[3] + X[i]*X[i]
    print(array)
    m = (N*array[0]-array[1]*array[2])/(N*array[3]-array[1]*array[1])
    b = (array[2] - m - array[1])/N
    
    return m,b

def LMSoffline2(X,t,N):
    w = 0
    b = 0

    eta = 0.1

    for epoch in range(100):
        dw = 0
        db = 0.0
        
        for i in range(N):
            # Prediction
            y = w * X[i] + b
            
            # LMS
            dw += (t[i] - y) * X[i]
            db += (t[i] - y)
            
        # Parameter updates
        w += eta * dw / N
        b += eta * db / N

        return w,b
    
    
N = 100
Features = 1
Noise = 15.0
X, t = make_regression(n_samples=N, n_features=1, noise=15.0)

m,b = LMSoffline(X,t,N)
PlotDataWithEQ(X,t,m,b)

m,b = LMSoffline2(X,t,N)
PlotDataWithEQ(X,t,m,b)



   




