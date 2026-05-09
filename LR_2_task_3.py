import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

names = [
    'sepal-length',
    'sepal-width',
    'petal-length',
    'petal-width',
    'class'
]

dataset = read_csv(url, names=names)

print("Shape:", dataset.shape)

print(dataset.head())
print(dataset.describe())
print(dataset.groupby('class').size())

dataset.plot(
    kind='box',
    subplots=True,
    layout=(2,2),
    figsize=(10,7),
    sharex=False,
    sharey=False
)

pyplot.suptitle("Box Plot for Iris Features", fontsize=14)
pyplot.tight_layout()
pyplot.show()

dataset.hist(figsize=(10,8), bins=15)

pyplot.suptitle("Histograms of Iris Dataset", fontsize=14)
pyplot.tight_layout()
pyplot.show()

scatter_matrix(
    dataset,
    figsize=(11,11),
    diagonal='hist'
)

pyplot.suptitle("Scatter Matrix", fontsize=14)
pyplot.show()

array = dataset.values
X = array[:,0:4]
y = array[:,4]

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1
)

models = []

models.append(('LR', LogisticRegression(
    solver='lbfgs',
    max_iter=200
)))

models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names_models = []

for name, model in models:

    kfold = StratifiedKFold(
        n_splits=10,
        random_state=1,
        shuffle=True
    )

    cv_results = cross_val_score(
        model,
        X_train,
        Y_train,
        cv=kfold,
        scoring='accuracy'
    )

    results.append(cv_results)
    names_models.append(name)

    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

pyplot.figure(figsize=(8,5))

pyplot.boxplot(results, labels=names_models)

pyplot.title("Algorithm Comparison", fontsize=14)
pyplot.xlabel("Algorithms")
pyplot.ylabel("Accuracy")

pyplot.grid(True)

pyplot.show()

model = SVC()

model.fit(X_train, Y_train)

predictions = model.predict(X_validation)

print("\nAccuracy:", accuracy_score(Y_validation, predictions))

print(
    "\nConfusion matrix:\n",
    confusion_matrix(Y_validation, predictions)
)

print(
    "\nReport:\n",
    classification_report(Y_validation, predictions)
)