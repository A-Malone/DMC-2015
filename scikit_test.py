from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.externals import joblib


clf = svm.SVC(gamma=0.001, C=100.)
iris = datasets.load_iris()
digits = datasets.load_digits()
print("{} items".format(len(digits.data)))
#X, y = iris.data, iris.target
a = clf.fit(digits.data[:-10], digits.target[:-10])
b = clf.predict(digits.data[-10])
print(b)

print(digits.target[:100])

#Plotting
plt.figure(1, figsize=(5, 5))
plt.imshow(digits.images[-10], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

#Saving a model to a file
#joblib.dump(clf, 'filename.pkl') 

#Loading a model from a file
#clf = joblib.load('filename.pkl')