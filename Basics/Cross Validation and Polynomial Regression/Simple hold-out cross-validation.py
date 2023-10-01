import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
import functions

from sklearn.model_selection import train_test_split

# to avoid warnings, pls ignore
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#dataset seperated to training(%70) and testing(%30)

N = 16
X, t = functions.create_dataset(N, noise=0.2)
X_train,X_test = train_test_split(X,test_size=0.3,train_size=0.7)
t_train,t_test = train_test_split(t,test_size=0.3,train_size=0.7)
print(X_train.shape)
print(X_test.shape)
x = np.linspace(0, 10, 100)#evenly scatter values 0-10 on 100
#train_test_split(array,0.3,0.7)

test_mse=[]
degrees = np.linspace(0,20,20)
for deg in degrees:

    w = np.polyfit(X_train,t_train,deg)
    y_test = np.polyval(w,X_test)

    mse = np.mean((t_test-y_test)**2)
    test_mse.append(mse)

plt.plot(degrees,test_mse)
plt.show()

#OBSERVATION:
#smallest errors are with degrees smaller then 7, optimal degree for this
#dataset whould be around 6-7