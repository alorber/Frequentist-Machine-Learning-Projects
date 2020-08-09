# Andrew Lorber
# Freq. ML Project 7 - Market Basket Analysis

# Apriori algorithms from https://github.com/pbharrin/machinelearninginaction3x/blob/master/Ch11/apriori.py

import numpy as np
import pandas as pd
import random

# Creates list of initial candidate sets (each containing 1 item)
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    # Frozen set to use it as a key in dict
    return list(map(frozenset, C1))

# For each candidate set, checks if support is greater than minSupport
# D = dataset, Ck = candidate set
def scanD(D, Ck, minSupport):
    # Counts num occurrences of each candidate set in data
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # Keeps sets based on support
    for key in ssCnt:
        # Calculates support
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

# Creates Ck
# Lk is previous list of subsets and k is new length of subsets
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            # Checks if first k-2 elements are equal
            if L1 == L2:
                # Adds union of sets to list
                retList.append(Lk[i] | Lk[j])
    return retList

# Main function to run
# Calculates all the subsets with a support greater than minSupport
def apriori(dataSet, minSupport=0.5):
    # Calculates first set of candidate sets from data
    C1 = createC1(dataSet)
    # Builds data
    D = list(map(set, dataSet))
    # Cuts out sets with support below minSupport
    L1, supportData = scanD(D, C1, minSupport)
    # List of valid subsets
    L = [L1]
    # Length of subsets
    k = 2
    while (len(L[k - 2]) > 0):
        # Calculates next set of candidate sets
        Ck = aprioriGen(L[k - 2], k)
        # Cuts out sets with support below minSupport
        Lk, supK = scanD(D, Ck, minSupport)
        # Adds support data to list
        supportData.update(supK)
        # Adds valid subsets to list
        L.append(Lk)
        k += 1
    return L, supportData

# Calculates the association rules
# L is list of valid subsets - output of apriori()
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    # Can only make rules with sets of two or more items
    for i in range(1, len(L)):
        for freqSet in L[i]:
            # H is list of right side of rule. X --> H[i]
            H1 = [frozenset([item]) for item in freqSet]
            # If more than 2 items in subset, calculates confidence with all possible H's
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# Calculates confidence of freqSet --> H[i]
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        # Calculates confidence
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            # print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        # Gets set of combinations of sets in H
        Hmp1 = aprioriGen(H, m + 1)
        # Calculates confidence
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # Can only merge rules if atleast two
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# Prints derived rules using item names instead of IDs
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item], end='; ')
        print("\n           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item], end=';')
        print("confidence: %f" % ruleTup[2])
        print("\n")

# ----------------------------------------------------------------------------------------------------

# Loads instacart data
df = pd.read_csv("/instacart_2017_05_01/order_products__prior.csv")

# Builds list of orders
orders = []
order_items = []
curr_order_num = 2
for order_item in np.array(df):
    if order_item[0] != curr_order_num:
        orders.append(order_items)
        order_items.clear()
        curr_order_num = order_item[0]
    order_items.append(order_item[1])
orders.append(order_items)

# Runs apriori algorithms on orders
L, supportData = apriori(orders)
rules = generateRules(L, supportData)

# Loads product IDs & names
product_df = pd.read_csv("/instacart_2017_05_01/products.csv")

# Builds dict to convert IDs to name
product_dict = {}
for product in np.array(product_df):
    product_dict[product[0]] = product[1]

# Prints rules using product names
# Too many rules to print, so picks random 50 to print
random.shuffle(rules)
pntRules(rules[:50], product_dict)


# ----------------------------------------------------------------------------------------------------


# Using MovieLens dataset
import io
from surprise import get_dataset_dir
from surprise import Dataset

# Loads move dataset
data = Dataset.load_builtin("ml-100k")

# Gets top 5 rated movies for each user
ratings = pd.DataFrame({"userID": [rating[0] for rating in data.raw_ratings],
                        "movieID": [rating[1] for rating in data.raw_ratings],
                        "Rating": [rating[2] for rating in data.raw_ratings]}) \
            .groupby(["userID"])

# Saves those that are 4 or 5 stars
topMovies = [[rating[1] for rating in np.array(ratings.get_group(str(uid)).sort_values(["Rating"],
                                                                                       ascending=False)
                                               .head(5)) if rating[2] > 3.0]
             for uid in list(ratings.groups)]

# Runs apriori algorithm on movies
L, supportData = apriori(topMovies)
rules = generateRules(L, supportData)

# Builds dict to convert IDs to names
file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
id_to_name = {}
with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
    for line in f:
        line = line.split('|')
        id_to_name[int(line[0])] = line[1]

pntRules(rules, id_to_name)
