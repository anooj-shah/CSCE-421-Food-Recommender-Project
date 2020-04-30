import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

recipes = pd.read_csv('data/core-data-train_rating.csv')
recipes = recipes[['user_id', 'recipe_id', 'rating']]
recipes.dropna(inplace = True) # drop any NaN if any
recipes = recipes[:50000]

coreRecipes = pd.read_csv('data/core-data_recipe.csv')
coreRecipes = coreRecipes[['recipe_id', 'recipe_name']]
coreRecipes.dropna(inplace = True)

combinedRecipes = pd.merge(recipes, coreRecipes, on = 'recipe_id')

combinedRecipes = combinedRecipes.drop_duplicates(['user_id', 'recipe_name'])

combinedRecipes = combinedRecipes.pivot(index = 'recipe_name', columns = 'user_id', values = 'rating').fillna(0)

recipeMatrix = csr_matrix(combinedRecipes.values)

# knn
recipeRecomm = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
recipeRecomm.fit(recipeMatrix)

# testing 
randomChoice = np.random.choice(combinedRecipes.shape[0])
print(combinedRecipes.shape[0])
distances, indices = recipeRecomm.kneighbors(combinedRecipes.iloc[randomChoice].values.reshape(1, -1), n_neighbors = 11)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print("Recommendations for Recipe: '{0}' on priority basis:\n".format(combinedRecipes.index[randomChoice]))
    else:
        print('{0}: {1}'.format(i, combinedRecipes.index[indices.flatten()[i]]))