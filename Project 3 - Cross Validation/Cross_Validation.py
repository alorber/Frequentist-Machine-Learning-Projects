# Andrew Lorber
# Freq. ML Project 3 - Cross Validation

import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest

# Creates features
N = 50  # Samples
p = 5000  # Predictors
X = np.random.normal(size=(N, p))

# Normalizes features
X = preprocessing.normalize(X)

# Creates Label
y = [0]*25 + [1]*25
np.random.shuffle(y)

# Part A - Wrong Way to do Cross Validation
# ------------------------------------------

# Selects features
selector = SelectKBest(k=100)
new_X = selector.fit_transform(X, y)

# Splits training data into 5 folds
num_per_fold = len(new_X) // 5
X_folds = [new_X[i * num_per_fold : (i+1) * num_per_fold] for i in range(5)]
y_folds = [y[i * num_per_fold : (i+1) * num_per_fold] for i in range(5)]

# Loops through folds
correct = 0
for k in range(5):
    # Removes fold from dataset
    X_k = np.concatenate([X_folds[i] for i in range(5) if (i != k)])
    y_k = np.concatenate([y_folds[i] for i in range(5) if (i != k)])

    # Builds a KNN classifier without the k_th fold
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_k, y_k)

    # Tests model on holdout fold
    y_predicted = knn.predict(X_folds[k])
    for i in range(len(X_folds[k])):
        if y_predicted[i] == y_folds[k][i]:
            correct += 1

print("\nPart A - Incorrect Cross Validation")
print("--------------------------------------")
print("\n ", correct)
print("-------   =  ", correct/50)
print(" ", 50)


# Part B - Correct Way to do Cross Validation
# --------------------------------------------

# Splits training data into 5 folds
num_per_fold = len(X) // 5
X_folds = [X[i * num_per_fold : (i+1) * num_per_fold] for i in range(5)]
y_folds = [y[i * num_per_fold : (i+1) * num_per_fold] for i in range(5)]

# Loops through folds
correct = 0
for k in range(5):
    # Removes fold from dataset
    X_k = np.concatenate([X_folds[i] for i in range(5) if (i != k)])
    y_k = np.concatenate([y_folds[i] for i in range(5) if (i != k)])

    # Selects features
    selector = SelectKBest(k=100)
    new_X_k = selector.fit_transform(X_k, y_k)

    # Builds a KNN classifier without the k_th fold
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(new_X_k, y_k)

    # Tests model on holdout fold
    y_predicted = knn.predict(selector.transform(X_folds[k]))
    for i in range(len(X_folds[k])):
        if  y_predicted[i] == y_folds[k][i]:
            correct += 1

print("\nPart B - Correct Cross Validation")
print("------------------------------------")
print("\n ", correct)
print("-------   =  ", correct/50)
print(" ", 50)
