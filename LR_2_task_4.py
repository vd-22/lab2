import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

input_file = 'income_data.txt'


X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 2000

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line.strip().split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoders = {}
X_encoded = np.empty(X.shape, dtype=object)

for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders[i] = le

X_final = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=5
)

models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=300)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=5)))
models.append(('NB', GaussianNB()))
models.append(('SVM', LinearSVC(random_state=5, max_iter=5000)))


results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")


best_model = DecisionTreeClassifier(random_state=5)
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, predictions, average='weighted') * 100, 2), "%")
print("Recall:", round(recall_score(y_test, predictions, average='weighted') * 100, 2), "%")
print("F1:", round(f1_score(y_test, predictions, average='weighted') * 100, 2), "%")
