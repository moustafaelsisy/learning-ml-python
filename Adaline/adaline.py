import numpy as np

class Adaline:
    def __init__(self, learnRate = 0.01, maxEpochs = 10):
        self.learnRate = learnRate
        self.maxEpochs = maxEpochs
        self.weights_ = None

    def fit(self, X, Y):
        self.weights_ = np.zeros(X.shape[1] + 1)
        #Apply gradient descent
        for _ in range(self.maxEpochs):
            errors = Y - self._getNetInputs(X)
            self.weights_[1:] += self.learnRate * X.T.dot(errors)
            self.weights_[0] += self.learnRate * errors.sum()

    def _getNetInputs(self, X):
        #Compute tranpose(weights) * X
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        netInputs = self._getNetInputs(X)
        return np.where(netInputs >= 0.0, 1, -1)

    def getWeights(self):
        return self.weights_
