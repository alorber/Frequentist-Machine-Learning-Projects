# Andrew Lorber
# Frequentist ML Project 2 - Logistic Regression

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(thetas, x):
    z = np.dot(thetas, x)
    return 1 / (1 + np.exp(-z))

# Calculates the likelihood of y given X & thetas
def likelihood(thetas, X, y):
    likelihood = 1
    for i in range(len(X)):
        if y[i] == 0:
            likelihood *= (1 - sigmoid(thetas, X[i]))
        else:
            likelihood *= sigmoid(thetas, X[i])
    return likelihood

# Calculates the log likelihood of y given X & thetas
def logLikelihood(thetas, X, y):
    log_likelihood = 0
    for i in range(len(X)):
        log_likelihood += (y[i] * np.log(sigmoid(thetas, X[i])) + ((1 - y[i]) * np.log(1 - sigmoid(thetas, X[i]))))

    return log_likelihood

# Calculates the derivative of the log likelihood with respect to theta_j (where j is a feature)
def d_log_likelihood(thetas, x, y, j):
    return (y - sigmoid(thetas, x)) * x[j]


# Banknote Authentication Dataset
# Features are characteristics of banknote images & label is Authentic (1) or Forged (0)
df = pd.read_csv("banknote_authentication.txt",
                 names=["Variance of WTI", "Skewness of WTI", "Curtosis of WTI", "Entropy of Image", "Authentic"])
                        # WTI = Wavelet Transformed Image

# Drops rows missing data
df.dropna(inplace=True)

# Normalizes features
X = np.array(df.drop(['Authentic'], axis=1))
X = preprocessing.normalize(X)

# Label
y = np.array(df['Authentic'])

# Splits data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val_test, y_val_test, test_size=0.5)


# Baseline Guess
correct = 0
for i in range(len(X_test)):
    if y_test[i] == 0:
        correct += 1

print("\nBaseline: Guessing Forged")
print(' ', correct)
print('-------     =     ', correct/len(X_test))
print(' ', len(X_test))


# Part A - Stochastic Gradient Descent Without Regularization
# ------------------------------------------------------------

thetas = [0]*len(X_train[0])
step = .1
no_reg_likelihood = [likelihood(thetas, X_test, y_test)]

# Loops through observations
for i in range(len(X_train)):
    # Loops through features
    for j in range(len(X_train[0])):
        # Updates theta
        thetas[j] += step * d_log_likelihood(thetas, X[i], y[i], j)
    # Calculates likelihood for plot
    no_reg_likelihood.append(likelihood(thetas, X_test, y_test))

# Scores model
correct = 0
for i in range(len(X_test)):
    if (round(sigmoid(thetas, X_test[i])) == y_test[i]):
        correct += 1

print("\nNo Regularization")
print(' ', correct)
print('-------     =     ', correct/len(X_test))
print(' ', len(X_test))


# Part B - Stochastic Gradient Descent With L2 Regularization
# ------------------------------------------------------------

best_model = [-1, -1, [], []]  # best_model = [lambda, #correct, thetas, likelihood]
step = .1

# Tests different lambdas
for lam in np.arange(0, .1, .01):
    thetas = [0] * len(X_train[0])
    l2_likelihood = [likelihood(thetas, X_test, y_test)]

    # Loops through observations
    for i in range(len(X_train)):
        # Loops through features
        for j in range(len(X_train[0])):
            # Updates theta
            thetas[j] += step * (d_log_likelihood(thetas, X[i], y[i], j) - (2 * lam * thetas[j]))
        # Calculates likelihood for plot
        l2_likelihood.append(likelihood(thetas, X_test, y_test))

    # Scores on validation set
    correct = 0
    for i in range(len(X_val)):
        if (round(sigmoid(thetas, X_val[i])) == y_val[i]):
            correct += 1

    if correct > best_model[1]:
        best_model = [lam, correct, thetas, l2_likelihood]

# Scores on test set
correct = 0
for i in range(len(X_test)):
    if (round(sigmoid(best_model[2], X_test[i])) == y_test[i]):
        correct += 1

print("\nL2 Regularization")
print("-----------------")
print(' ', correct)
print('-------     =     ', correct/len(X_test))
print(' ', len(X_test))

print('\nLambda: ', best_model[0])

# Plots likelihood without regularization
plt.plot(range(len(no_reg_likelihood)), no_reg_likelihood)
# Plots likelihood with L2 regularization
plt.plot(range(len(best_model[3])), best_model[3])
plt.title("Likelihood Plot")
plt.xlabel("Iteration")
plt.ylabel("Likelihood")
plt.legend(["Unregularized", "L2 Regularization"])
plt.show()

