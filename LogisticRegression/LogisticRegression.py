import numpy as np

class LogisticRegression:
    def __init__(self, learnRate = 0.01, maxEpochs = 10):
        self.learnRate = learnRate
        self.maxEpochs = maxEpochs
        self.weights_ = None

    def fit(self, X, Y):
        self.weights_ = np.zeros(X.shape[1] + 1)
        for _ in range(self.maxEpochs):
            errors = Y - self.activation(X)
            self.weights_[1:] += self.learnRate * X.T.dot(errors)
            self.weights_ += self.learnRate * errors.sum()

    def activation(self, X):
        return 1/(1+np.exp(-self.netInput(X)))

    def netInput(self, X):
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        #Return the prediction (and its probability) for each test observation
        return np.apply_along_axis(self._predict, 1, X)

    def _predict(self, x):
        activation = self.activation(x)
        prediction = np.where(activation >= 0.5 , 1, 0)
        #Select between probability of being classified as 1, and probability
        #of being classified as 0 based on which classification is more likely
        #and return its respective probability
        probability = np.where(activation >= 0.5, activation, 1 - activation)

        return [prediction, probability]
