import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
import functions

#inputs X, desired outputs t, desired degree deg
#w = np.polyfit(X,t,deg)

#predict values
#y=np.polyval(w,x)

# to avoid warnings, pls ignore
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def mseCalculation(X,t):
    training_mse=[]
    degrees= []
    for i in range(20):
        Deg = i
        w = np.polyfit(X, t, Deg)
        y = np.polyval(w, X)
        mse = np.mean((t-y)**2)
        training_mse.append(mse)
        degrees.append(Deg)
        print("Degree: ",Deg,"TRaining error",mse)

    return training_mse,degrees

N = 16
X, t = functions.create_dataset(N, noise=0.2)

x = np.linspace(0, 10, 100)#evenly scatter values 0-10 on 100
#plt.plot(x,x*np.sin(x),label = "Ground Truth")
#plt.scatter(X,t)
#plt.show()

deg = 20
w = np.polyfit(X,t,deg)
y = np.polyval(w,x)

print(y.shape)

plt.plot(x,x*np.sin(x),label = "Polynomial Regression")
plt.plot(x,y)
plt.scatter(X,t)
plt.show()

#OBSERVATION:
#after a degree of 10, line to starts to overfit

#mean square error change over a range of degree between 1-20
training_mse,degrees = mseCalculation(X,t)

plt.plot(degrees,training_mse)
plt.show()
#OBSERVATION:
#degrees smaller than 6 are have larger training error.



