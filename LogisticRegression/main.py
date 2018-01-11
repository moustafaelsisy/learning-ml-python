from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from LogisticRegression import LogisticRegression
import numpy as np

def main():
    iris = datasets.load_iris()
    scaler = StandardScaler()
    lr = LogisticRegression()

    X_all = iris.data[0:100]
    Y_all = iris.target[0:100] #Setosa: 0, Versicolor: 1

    #Standardize
    scaler.fit(X_all)
    X_all = scaler.transform(X_all)

    #Split the data into training (70%) and test (30%) data
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size = 0.3)

    #Train the LogisticRegression classifier
    lr.fit(X_train, Y_train)

    #Make predictions
    predictions = lr.predict(X_test)
    print(predictions) #prints [prediction, probability]
    print( "Accuracy: {:.2f}".format(accuracy_score(Y_test, predictions[:,0])) )

main()
