import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

#generate dataset
def generateDataset(S,F,N):
   return make_regression(S,F,N) 

#draw the graphic of dataset
def showGraphic(X,t):
    print(X.shape)
    print(t.shape)
    plt.figure(figsize=(10, 5))
    plt.scatter(X, t)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()


#least mean squares
def LMS():
    pass 

sampleSize = 100
Features = 1
Noise = 15.0
X,t = generateDataset(sampleSize,Features,Noise)
showGraphic(X,t)





