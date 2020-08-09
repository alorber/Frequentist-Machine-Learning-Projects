# Andrew Lorber
# Freq. ML Project 6 - NMF

import io
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import NMF, Dataset, accuracy, get_dataset_dir
from surprise.model_selection import GridSearchCV

# Loads move dataset
data = Dataset.load_builtin("ml-100k")

# Gets ratings
raw_ratings = data.raw_ratings

# shuffles ratings
random.shuffle(raw_ratings)

# Splits ratings into training and testing sets
amt_test = .2
train_raw_ratings = raw_ratings[int(amt_test * len(raw_ratings)):]
test_raw_ratings = raw_ratings[:int(amt_test * len(raw_ratings))]

# Uses training set
data.raw_ratings = train_raw_ratings

# Finds best parameters for NMF model with bias
# Scores using MSE
params = {"biased": [True], "n_factors": np.arange(2,12,2)}
nmf = GridSearchCV(NMF, params, measures=["mse"], cv=3)
nmf.fit(data)

print("\nBest number of factors found:", nmf.best_params['mse']['n_factors'])

# Trains NVM using best parameters found
best_nmf = NMF(biased=True, n_factors=nmf.best_params['mse']['n_factors'])
best_nmf.fit(data.build_full_trainset())

# Tests on training set
predictions = best_nmf.test(data.build_full_trainset().build_testset())
mse = accuracy.mse(predictions, verbose=False)
print("Training Set MSE:", mse)

# Scores on test set
predictions = best_nmf.test(data.construct_testset(test_raw_ratings))
mse = accuracy.mse(predictions, verbose=False)
print("Test Set MSE:", mse)


# Checks Recommendations
# -----------------------
recs = defaultdict(list)  # List of recommendations for each user
num_recs = 5    # Number of recommendations to get for each user
for uid, iid, true_r, est, _ in predictions:
    recs[uid].append((iid, est))

# Sorts the predictions for each user and retrieve the num_recs highest ones.
for uid, user_ratings in recs.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    recs[uid] = user_ratings[:num_recs]

# Builds dict to convert IDs to names
file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
rid_to_name = {}
name_to_rid = {}
with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
    for line in f:
        line = line.split('|')
        rid_to_name[line[0]] = line[1]
        name_to_rid[line[1]] = line[0]


# Gets top 5 rated movies for each user
ratings = pd.DataFrame({"userID": [rating[0] for rating in data.raw_ratings],
                        "movieID": [rating[1] for rating in data.raw_ratings],
                        "Rating": [rating[2] for rating in data.raw_ratings]}) \
            .groupby(["userID"])

# Prints movie recommendations for users
for uid, user_ratings in list(recs.items())[:10]:
    print("\nUser:", uid)
    print("-----------")
    top5 = np.array(ratings.get_group(str(uid)).sort_values(["Rating"], ascending=False).head(5))
    print("Top Rated:", [rid_to_name[movieID] for (_, movieID, _) in top5])
    print("Recommended:", [rid_to_name[iid] for (iid, _) in user_ratings])




