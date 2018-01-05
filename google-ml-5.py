"""
This script follows through with Josh Gordon's Machine Learning Recipes series,
available at https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal .
The principals covered in the fifth entry of the series have been implemented here on the iris dataset
"""
def getDistance(a, b):
    from scipy.spatial import distance
    return distance.euclidean(a,b)

#Write our own basic KNN classifier
class MyClassifier:
    def fit(self, x, y):
        self.trainX = x
        self.trainY = y

    def predict(self, data):
        predictions = []
        for row in data:
            #Initialise best predictor to be the first training entry
            bestDistance = getDistance(self.trainX[0], row)
            bestPrediction = self.trainY[0]
            #Look for the best predictor
            for i in range(1, len(self.trainX)):
                distance = getDistance(self.trainX[i], row)
                if(distance < bestDistance):
                    distance = bestDistance
                    bestPrediction = self.trainY[i]
            predictions.append(bestPrediction)

        return predictions


def main():
    from sklearn import datasets
    from sklearn.cross_validation import train_test_split
    iris = datasets.load_iris()

    trainX, testX, trainY, testY = train_test_split(iris.data, iris.target, test_size = 0.5)

    classifier = MyClassifier()
    classifier.fit(trainX, trainY)
    predictions = classifier.predict(testX)

    from sklearn.metrics import accuracy_score
    print( "Accuracy: {:.2f}".format(accuracy_score(testY, predictions)) )

main()
