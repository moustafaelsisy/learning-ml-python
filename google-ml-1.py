"""
This script follows through with Josh Gordon's Machine Learning Recipes series,
available at https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal .
The principals covered in the first entry of the series have been implemented here on
a different custom problem, in order to implement a decision tree classifier
"""
from sklearn import tree

#Conducts Electricity, Shiny, Melting point, boiling point, Sonorous
data = [[True, True, 2345, 4300, True], [True, True, -20, 270, True], [True, True, 1500, 2300, True],
        [True, False, 800, 1400, False], [False, False, 300, 560, False], [False, False, -300, -264, False],
        [False, False, 60, 120, False]]

labels = ["Metal", "Metal", "Metal", "Metalloid", "Non-Metal", "Non-Metal", "Non-Metal"]

clf = tree.DecisionTreeClassifier()
clf.fit(data, labels)
#Get prediction for Mercury
print(clf.predict([[True, True, -38.83, 356.7, True]])) #Metal
