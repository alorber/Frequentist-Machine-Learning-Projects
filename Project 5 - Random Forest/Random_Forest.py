# Andrew Lorber
# Freq. ML Project 5 - Random Forest

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Spambase dataset
# Features are email characteristics & label is Spam (1) or Not Spam (0).
df = pd.read_csv("/Spam/spambase.data")

# Drops rows missing data
df.dropna(inplace=True)

# Features
X = np.array(df.drop(["Spam"], axis=1))
X = preprocessing.normalize(X)

# Label
y = np.array(df["Spam"])

# Splits data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val_test, y_val_test, test_size=0.5, shuffle=True)

# Baseline Accuracy (Default Parameters)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_predicted = rf.predict(X_test)
correct = 0
for i in range(len(y_test)):
    if y_test[i] == y_predicted[i]:
        correct += 1

print("\nBaseline Accuracy:")
print("---------------------")
print("\n ", correct)
print("-------   =  ", correct/len(y_test))
print(" ", len(y_test))

# Tests different forest sizes
best_model = [-1, -1]  # best_model = [# trees, # correct]
for n in np.arange(20, 200, 5):
    rf = RandomForestClassifier(n_estimators=n)
    rf.fit(X_train, y_train)

    # Tests on validation set
    y_predicted = rf.predict(X_val)
    correct = 0
    for i in range(len(y_val)):
        if y_val[i] == y_predicted[i]:
            correct += 1

    # Updates best_model
    if correct > best_model[1]:
        best_model = [n, correct]


# Builds best model
rf = RandomForestClassifier(n_estimators=best_model[0])
rf.fit(X_train, y_train)

# Tests on test set
y_predicted = rf.predict(X_test)
correct = 0
for i in range(len(y_test)):
    if y_test[i] == y_predicted[i]:
        correct += 1


print("\nBest Forest Size: ", best_model[0])
print("\n ", correct)
print("-------   =  ", correct/len(y_test))
print(" ", len(y_test))

# Plots feature importance
labels = ["make","address","all","3d","our","over","remove","internet","order","mail","receive","will","people",
          "report","addresses","free","business","email","you","credit","your","font","000","money","hp","hpl",
          "george","650","lab","labs","telnet","857","data","415","85","technology","1999","parts","pm","direct",
          "cs","meeting","original","project","re","edu","table","conference",";","(","[","!","$","#",
          "capital_average","capital_longest","capital_total"]
plt.bar(labels, rf.feature_importances_)
plt.show()
