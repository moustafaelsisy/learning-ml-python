from adaline import Adaline
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def loadIris():
    iris = datasets.load_iris()
    #Get Setosa and Versicolor Sepal Length and Petal Length, respectively
    x = pd.DataFrame(iris.data).iloc[0:100, [0,2]].values
    y = np.where(iris.target[0:100] == 0, -1, 1) #{Setosa => -1, Versicolor => 1}
    return x, y

def standardize(X):
    stdX = np.copy(X)
    for col in range(X.shape[1]):
        stdX[:, col] = (X[:, col] - X[:, col].mean())/X[:, col].std()
    return stdX

def plotIris(X, Y):
    plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
    plt.scatter(X[50:100, 0], X[50:100, 1], color="green", marker="o", label="versicolor")
    plt.xlabel("sepal length(cm)")
    plt.ylabel("petal length(cm)")
    plt.legend(loc="upper left")

def plotDecisionBoundary(classifier, X, Y):
    plotIris(X, Y)
    weights = classifier.getWeights()
    #Plot the line connecting y at smallest and largest x in the setosa and versicolor iris dataset
    #by rearranging the net input equation for the precepetron
    x1 = X[:,0].min()
    x2 = X[:,0].max()
    y1 = np.dot(weights[:2], [-1, -x1])/weights[2]
    y2 = np.dot(weights[:2], [-1, -x2])/weights[2]

    plt.plot([x1, x2], [y1, y2])
    plt.savefig("decision_boundary.png")

def main():

    X,Y = loadIris()
    stdX = standardize(X)

    classifier = Adaline(learnRate = 0.01, maxEpochs = 20)
    classifier.fit(stdX, Y)
    plotDecisionBoundary(classifier, stdX, Y)

main()
