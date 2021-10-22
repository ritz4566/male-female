from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

X = [[181,63,40],[177,40,55],[165,54,62],[187,48,56],[178,45,67],[169,45,85],[177,70,40],[171,75,42],[190,90,47],[154,54,27]]

Y = ['male','female','felmale','male','female','male','female','male','male','femmale']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[187,45,98]])

print(prediction)
clf1 = svm.SVC(probability=True)

clf1 = clf1.fit(X,Y)

prediction_2 = clf1.predict([[187,45,98]])

print(prediction_2)

clf2 = KNeighborsClassifier(n_neighbors=3)

clf2 = clf2.fit(X,Y)

prediction_3 = clf2.predict([[187,45,98]])

print(prediction_3)

clf3 = GaussianProcessClassifier()

clf3 = clf3.fit(X,Y)

prediction_4 = clf3.predict([[187,45,98]])

print(prediction_4)

clf4 = MLPClassifier(learning_rate='constant', learning_rate_init=0.001)

clf4 = clf4.fit(X,Y)

prediction_5 = clf4.predict([[187,45,98]])

print(prediction_5)

clf5 = AdaBoostClassifier()

clf5 = clf5.fit(X,Y)

prediction_6 = clf5.predict([[187,45,98]])

print(prediction_6)