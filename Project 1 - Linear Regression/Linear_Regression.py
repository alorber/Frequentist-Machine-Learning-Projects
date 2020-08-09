# Andrew Lorber
# Frequentist ML Project 1 - Linear Regression

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.linear_model import Lasso, lasso_path
import matplotlib.pyplot as plt

# Calculates the residual sum of squares
def calculateRSS(y_true, y_predicted):
    rss = 0
    for i in range(len(y_true)):
        rss += (y_true[i] - y_predicted[i]) ** 2

    return rss

# Calculates R^2 of line
def calculateR2(y_true, y_avg, rss):
    true_RSS = 0
    for i in range(len(y_true)):
        true_RSS += (y_true[i] - y_avg) ** 2

    return 1 - (rss / true_RSS)

# Combined Cycle Power Plant Dataset from UCI Database
# Features are power plant conditions and label is energy output
df = pd.read_excel("/CCPP/Energy.xlsx")

# Drops rows missing data
df.dropna(inplace=True)

# Renames columns
df.rename(columns={'AT': 'Temperature', 'AP': 'Ambient Pressure', 'RH': 'Relative Humidity',
                   'V': 'Exhaust Vacuum', 'PE': 'Energy Output'},
          inplace=True)

# Normalizes features
X = np.array(df.drop(["Energy Output"], axis=1))
X = preprocessing.normalize(X)

# Label
y = np.array(df["Energy Output"])

# Splits data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val_test, y_val_test, test_size=0.5)


# Calculates a baseline MSE
avg = np.average(y_test)
rss = calculateRSS(y_test, [avg]*len(y_test))
print('\nBaseline MSE: ', rss / len(y_test))


# Part A - Plain Linear Regression
# --------------------------------

# Inserts a 1 for the intercept B_0
X_reg_train = np.insert(X_train, 0, [1]*len(X_train), axis=1)
X_reg_test = np.insert(X_test, 0, [1]*len(X_test), axis=1)

# Calculates betas
betas = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_reg_train),
                                                    X_reg_train)),
                            np.transpose(X_reg_train)),
                    y_train)

# Predicts outputs for the testing set
y_predicted = np.matmul(X_reg_test, betas)

# Calculates Mean Squared Error
rss = calculateRSS(y_test, y_predicted)
mean_squared_error = rss / len(y_test)

# Calculates R^2 of line
y_avg = sum(y_test) / len(y_test)
r2 = calculateR2(y_test, y_avg, rss)

print("\nRegular Linear Regression")
print("-------------------------")
print("MSE: ", mean_squared_error)
print("R^2: ", r2)


# Part B - Ridge Regression
# --------------------------------

X_ridge_train = np.insert(X_train, 0, [1]*len(X_train), axis=1)
X_ridge_val = np.insert(X_val, 0, [1]*len(X_val), axis=1)
X_ridge_test = np.insert(X_test, 0, [1]*len(X_test), axis=1)

# Searches for best lambda
min = [-1, np.inf, []]  # min = [lambda, rss, betas]
# Calculates betas for different lambdas
for lam in np.arange(0, .6, .001):
    betas_ridge = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_ridge_train),
                                                               X_ridge_train)
                                                    + lam * np.identity(len(X_ridge_train[0]))) ,
                                    np.transpose(X_ridge_train)),
                            y_train)

    # Calculates predicted output for validation set
    y_predicted = np.matmul(X_ridge_val, betas_ridge)

    # Calculates RSS
    rss = calculateRSS(y_val, y_predicted)

    # If lower RSS, update best lambda
    if rss < min[1]:
        min = [lam, rss, betas_ridge]

# Once best lambda has been found, tests on test set
y_predicted = np.matmul(X_ridge_test, min[2])

# Calculates Mean Squared Error
rss = calculateRSS(y_test, y_predicted)
mean_squared_error = rss / len(y_test)

# Calculates R^2 of line
r2 = calculateR2(y_test, y_avg, rss)

print("\nRidge Regression")
print("-------------------------")
print("Lambda: ", min[0])
print("MSE: ", mean_squared_error)
print("R^2: ", r2)


# Part C - Lasso Regression
# --------------------------------

best = Lasso(alpha=.001, max_iter=8000)
best.fit(X_train, y_train)
best_lam = .001

for lam in np.arange(.002, .6, .001):
    lasso = Lasso(alpha=lam, max_iter=8000)
    lasso.fit(X_train, y_train)
    if lasso.score(X_val, y_val) > best.score(X_val, y_val):
        best = lasso
        best_lam = lam


y_predicted = best.predict(X_test)
rss = calculateRSS(y_test, y_predicted)
mean_squared_error = rss / len(y_test)

print("\nLasso Regression")
print("-------------------------")
print("Lambda: ", best_lam)
print("MSE: ", mean_squared_error)
print("R^2: ", best.score(X_test, y_test))

# Lasso Plot
alphas, lasso_betas, _ = lasso_path(X_train, y_train, alphas=np.arange(.01, .5, .01))
plt.plot(alphas, lasso_betas[0])
plt.plot(alphas, lasso_betas[1])
plt.plot(alphas, lasso_betas[2])
plt.plot(alphas, lasso_betas[3])

# Marks chosen lambda
plt.plot([best_lam, best_lam], [np.min(lasso_betas), np.max(lasso_betas)], '--')

# Labels
plt.title("Lasso Plot")
plt.xlabel("Lambdas")
plt.ylabel("Betas")
plt.legend(np.append(df.columns[:-1], ["Chosen Lambda"]))

plt.show()


