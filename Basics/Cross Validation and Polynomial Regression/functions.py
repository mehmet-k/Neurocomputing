import numpy as np
def create_dataset(N, noise):
    #Creates a dataset of N points generated from x*sin(x) plus some noise.
    
    x = np.linspace(0, 10, 300)
    rng = np.random.default_rng()
    rng.shuffle(x)
    x = np.sort(x[:N])
    t = x * np.sin(x) + noise*rng.uniform(-1.0, 1.0, N)
    
    return x, t