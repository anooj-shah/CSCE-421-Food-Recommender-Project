import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

recipes = pd.read_csv('data/core-data-train_rating.csv')
recipes = recipes[['user_id', 'recipe_id', 'rating',]]
recipes.dropna(inplace = True) # drop any NaN if any

print("begin")
# recipes_rating = (recipes.
#     groupby(by = ['title'])['rating'].
#     count().
#     reset_index())


# for i in range(0, len(data.index)):
#     split_data=re.split(r'[,]', data.iloc[i, 6])
#     for k,l in enumerate(split_data):
#         split_data[k]=(split_data[k].replace("['", ""))
#         split_data[k]=(split_data[k].replace(" '", ""))
#         split_data[k]=(split_data[k].replace("'", ""))
#         split_data[k]=(split_data[k].replace("]", ""))
#     split_data=','.join(split_data[:])
#     data['categories'].iloc[i]=split_data

# print(data.head(5))

# get user input for what categories they want to filter by
# categoryInput = input("Please enter categories (for multiple, enter | in between each category): ")

# reduce the amount of entries in the dataframe
# data = data[data['categories'].str.contains(categoryInput)]
recipes = recipes.drop_duplicates(['user_id', 'recipe_id'])
print("dropped")

recipes = recipes.pivot(index = 'recipe_id', columns = 'user_id', values = 'rating').fillna(0)

print("end?")

dataMatrix = csr_matrix(recipes.values)