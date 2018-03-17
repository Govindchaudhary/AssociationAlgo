# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
    
'''1) support says that how much % of a particular combn of items are present in the dataset
for eg. how much % of people bought a combination of both A and B and may be more(we are assuming min=2)
we calculate it by (no. of time a particular combn of items purchased/total no. of transactions)
for eg. if a combn prodt is bought by 3 customers per day
then min_support can be 3*7(for a week)/7500 =.003
meaning that we only include that combn in our rules which are bought by minm 3 customers a day
or support >.003

2)confidence says that if a customer buy a product A then what is the probablity that he also buy B
    we say min_confidence=.2(20%) meaning that rules contain only those combn of A and B or may be more(we are assuming min=2)
    for which if A is bought then probablity of B being bought > 20%
    
3)lift represents the relevance of our rules ie. how relevently  A and B or may be more(we are assuming min=2) are related to each other

'''

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)