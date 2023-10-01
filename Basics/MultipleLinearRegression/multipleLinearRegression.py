import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

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
        plt.title(dataset.feature_names[i])
    plt.show()

if __name__ == '__main__':

    dataset = fetch_california_housing()
    X = dataset.data
    t = dataset.target

    print(X.shape)
    print(t.shape)
    print(dataset.DESCR)

    #PlotAllDataIndividualy(X,t)

    #make LinearRegression

    #calculations without normalization
    reg = LinearRegression()
    reg.fit(X,t)
    y = reg.predict(X)

    #calculations with normalization
    X_normalized = (X - X.mean(axis=0))/X.std(axis=0)
    reg.fit(X_normalized,t)
    z=reg.predict(X_normalized)
    
    p1=Process(target=PlotDataAndPredictions,args=(X,t,y))
    p2=Process(target=PlotDataAndPredictions,args=(X,t,z))

    p1.start()
    p2.start()

    plt.figure(figsize=(12, 5))
    plt.plot(np.abs(reg.coef_))
    plt.xlabel("Feature")
    plt.ylabel("Weight")
    plt.show()