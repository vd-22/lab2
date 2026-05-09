import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

input_file = 'income_data.txt'

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

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


classifier = OneVsOneClassifier(
    SVC(kernel='poly', degree=8, random_state=0)
)

classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1:", round(f1 * 100, 2), "%")


cv_f1 = cross_val_score(
    OneVsOneClassifier(
        SVC(kernel='poly', degree=8, random_state=0)
    ),
    X_final, y,
    scoring='f1_weighted',
    cv=3
)

print("Cross-val F1:", round(cv_f1.mean() * 100, 2), "%")


input_data = [
    '37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
    'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
    '0', '0', '40', 'United-States'
]

input_data_encoded = []

for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded.append(int(item))
    else:
        input_data_encoded.append(
            int(label_encoders[i].transform([item])[0])
        )

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

predicted_class = classifier.predict(input_data_encoded)

target_encoder = label_encoders[X.shape[1] - 1]

print(
    "Predicted class:",
    target_encoder.inverse_transform(predicted_class)[0]
)