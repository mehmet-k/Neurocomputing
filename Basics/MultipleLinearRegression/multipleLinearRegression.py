import numpy as np
import matplotlib.pyplot as plt

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


dataset = fetch_california_housing()
X = dataset.data
t = dataset.target

print(X.shape)
print(t.shape)
print(dataset.DESCR)

#PlotAllDataIndividualy(X,t)

#make LinearRegression
reg = LinearRegression()
reg.fit(X,t)
y = reg.predict(X)

mse = np.mean((t-y)**2)
print(mse)


PlotDataAndPredictions(X,t,y)