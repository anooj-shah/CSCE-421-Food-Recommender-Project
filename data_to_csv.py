import pandas as pd
import numpy as np
import json
import warnings
import csv
warnings.filterwarnings('ignore')


recipes = pd.read_csv('data/core-data-train_rating.csv')
recipes = recipes[['user_id', 'recipe_id', 'rating']]
recipes.dropna(inplace = True) # drop any NaN if any

coreRecipes = pd.read_csv('data/core-data_recipe.csv')
coreRecipes = coreRecipes[['recipe_id', 'recipe_name', 'ingredients']]
coreRecipes.dropna(inplace = True)

combinedRecipes = pd.merge(recipes, coreRecipes, on = 'recipe_id')

compression_opts = dict(method = 'zip', archive_name = 'combinedRecipes.csv')
combinedRecipes.to_csv('combinedRecipes.zip', index = False, compression = compression_opts)

