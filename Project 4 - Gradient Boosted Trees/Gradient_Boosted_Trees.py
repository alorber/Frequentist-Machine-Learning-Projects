# Andrew Lorber
# Freq. ML Project 4 - Gradient Boosted Trees

import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
import xgboost as xgb

# Spambase dataset
# Features are email characteristics & label is Spam (1) or Not Spam (0).
df = pd.read_csv("/Spam/spambase.data")

# Drops rows missing data
df.dropna(inplace=True)

# Features
X = np.array(df.drop(["Spam"], axis=1))

# Label
y = np.array(df["Spam"])

# Splits data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val_test, y_val_test, test_size=0.5, shuffle=True)

# Calculates Baseline Accuracy Using Default Parameters
gbt = xgb.XGBClassifier()
gbt.fit(X_train, y_train)
y_predicted = [round(y) for y in gbt.predict(X_test)]
correct = 0
for i in range(len(y_test)):
    if y_test[i] == y_predicted[i]:
        correct += 1

print("\n ", correct)
print("-------   =  ", correct/len(y_test))
print(" ", len(y_test))

# Tests different learning rates and max tree depths
best_model = [0, 0, -1]  # best_model = [alpha, depth, # correct]
for alpha in np.arange(0.05, 1, .05):
    for depth in np.arange(5, 12, 1):
        gbt = xgb.XGBClassifier(learning_rate=alpha, max_depth=depth)
        gbt.fit(X_train, y_train)

        # Tests on validation set
        y_predicted = [round(y) for y in gbt.predict(X_val)]
        correct = 0
        for i in range(len(y_val)):
            if y_val[i] == y_predicted[i]:
                correct += 1

        # Updates best_model
        if correct > best_model[2]:
            best_model = [alpha, depth, correct]

# Builds best model
gbt = xgb.XGBClassifier(learning_rate=best_model[0], max_depth=best_model[1])
gbt.fit(X_train, y_train)

# Tests on test set
y_predicted = [round(y) for y in gbt.predict(X_test)]
correct = 0
for i in range(len(y_test)):
    if y_test[i] == y_predicted[i]:
        correct += 1

print("\nBest alpha: ", best_model[0])
print("Best depth: ", best_model[2])
print("\n ", correct)
print("-------   =  ", correct/len(y_test))
print(" ", len(y_test))

# Plots feature importance
labels = ["make","address","all","3d","our","over","remove","internet","order","mail","receive","will","people",
          "report","addresses","free","business","email","you","credit","your","font","000","money","hp","hpl",
          "george","650","lab","labs","telnet","857","data","415","85","technology","1999","parts","pm","direct",
          "cs","meeting","original","project","re","edu","table","conference",";","(","[","!","$","#",
          "capital_average","capital_longest","capital_total"]
plt.bar(labels, gbt.feature_importances_)
plt.show()
