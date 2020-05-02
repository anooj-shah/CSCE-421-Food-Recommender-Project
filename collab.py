import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Retrieve the data
combinedData = pd.read_csv('data/combinedRecipes.csv')
combinedData.dropna(inplace = True) # Drop any rows that contain NaN
combinedData = combinedData[:50000] # Take in only the first 50,000 data

# Drop any duplicates of user_id and recipe_name
combinedData = combinedData.drop_duplicates(['user_id', 'recipe_name'])

# Pivot the data around the recipe_name and the user_id along with its ratings
combinedData = combinedData.pivot(index = 'recipe_name', columns = 'user_id', values = 'rating').fillna(0)

# Convert the pivot table into a matrix
recipeMatrix = csr_matrix(combinedData.values)

# knn
recipeRecomm = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
recipeRecomm.fit(recipeMatrix)

# testing 
randomChoice = np.random.choice(combinedData.shape[0])
distances, indices = recipeRecomm.kneighbors(combinedData.iloc[randomChoice].values.reshape(1, -1), n_neighbors = 11)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print("Recommendations for Recipe: '{0}' on priority basis:\n".format(combinedData.index[randomChoice]))
    else:
        print('{0}: {1}'.format(i, combinedData.index[indices.flatten()[i]]))