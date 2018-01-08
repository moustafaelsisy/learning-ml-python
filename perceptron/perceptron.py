import numpy as np

class Perceptron:
    def __init__(self, learnRate = 0.01,  learnAccuracy = 0.9, maxLearnEpochs = 10):
        self.learnRate = learnRate
        self.learnAccuracy = learnAccuracy
        self.maxLearnEpochs = maxLearnEpochs
        self.weights_ = None
        self._wException = Exception("Perceptron has not been trained!")

    def fit(self, X, y):
        self.weights_ = np.zeros(1 + X.shape[1])
        observations = X.shape[0]
        mistakes = observations #for 100% initial inaccuracy
        epochs = 0

        while epochs <= self.maxLearnEpochs and mistakes/observations > 1-self.learnAccuracy:
            mistakes = 0
            for xi, yi in zip(X, y):
                adjustment = self.learnRate * (yi - self.predict(xi))
                self.weights_[1:] += adjustment * xi
                self.weights_[0] += adjustment
                mistakes += 0 if adjustment == 0.0 else 1
            epochs += 1

        return self

    def predict(self, X):
        if(self.weights_ is None):
            raise self._wException

        netInput = np.dot(self.weights_[1:], X) + self.weights_[0]
        return 1 if netInput >= 0.0 else -1

    def getWeights(self):
        if(self.weights_ is None):
            raise self._wException

        return self.weights_
