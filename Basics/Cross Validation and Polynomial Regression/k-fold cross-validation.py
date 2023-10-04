import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process

import functions

from sklearn.model_selection import KFold

#split dataset to subsets, use each one to train 
# and others to test


if __name__ == '__main__':

    N = 16
    X, t = functions.create_dataset(N, noise=0.2)

    X_normalized = (X - X.mean(axis=0))/X.std(axis=0)
    
    k = N #when k = N it is called "leave one out cross validation",
          #more stable but also more expensive
    kf = KFold(n_splits=k,shuffle=True) 
    x = np.linspace(0, 10, 100)
    
    test_mse=[]

    Degrees = range(1,20)
    for train_index, test_index in kf.split(X, t):
        
        split_mse = []

        for Deg in Degrees:
            w = np.polyfit(X[train_index],t[train_index],Deg)
            y = np.polyval(w,X[test_index])

            mse = np.mean((t[test_index]-y)**2)
            split_mse.append(mse)
        
        test_mse.append(split_mse)
        
        print("Train:", train_index)
        print("Test:", test_index)
        print('-------')
    
    #plt.scatter(Degrees,test_mse)
    #plt.show()

    test_mse_mean = test_mse = np.mean(test_mse, axis=0)
    plt.plot(Degrees,test_mse_mean)
    plt.show()

    #OBSERVATION: 
    #Polynomial degree between 6-7 still seems to be the best resultç
    #increasing k value results in learning better but more expensive