from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import fetch_california_housing
import time

#1. MLR With L2 regularization aka RIDGE regression
#2. MLR With L1 regularization aka LASSO regression

def PlotAllDataIndividualy(a,b):
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.scatter(a[:,i],b)
        plt.title(dataset.feature_names[i])
    plt.show()

def PlotDataAndPredictions(a,b,y):
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.scatter(a[:,i],b)
        plt.scatter(a[:,i],y)
        plt.title
    plt.show()

def PlotFeatureWeight(reg):
    plt.plot(np.abs(reg.coef_))
    plt.xlabel("Feature")
    plt.ylabel("Weight")
    plt.show()

if __name__ == '__main__':
    
    dataset = fetch_california_housing()
    X = dataset.data
    t = dataset.target
    X_normalized = (X - X.mean(axis=0))/X.std(axis=0)

    #smaller alpha values work better for Lasso
    reg = Lasso(alpha=0.1)
    reg.fit(X_normalized,t)
    y = reg.predict(X_normalized)
    
    p1=Process(target=PlotDataAndPredictions,args=(X_normalized,t,y))
    p1.start()

    PlotFeatureWeight(reg)
    p1.terminate()  
    
    #greater alpha values work better for Ridge
    reg = Ridge(alpha=10)
    reg.fit(X_normalized,t)
    y = reg.predict(X_normalized)

    PlotDataAndPredictions(X,t,y)
    PlotFeatureWeight(reg)
