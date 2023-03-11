import random
import numpy as np

class RandomModel:
    def __init__(self):
        pass
    
    def fit(self, X_train, labels):
        self.train = X_train
        self.labels = labels

    def predict(self, test):
        out = np.zeros(len(test))
        for num in range(0,len(test)):
            out[num] = random.randint(1,9)
        return out
        
        