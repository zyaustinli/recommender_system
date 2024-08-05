import numpy as np
import pandas as pd

movie_names = pd.read_csv('movies.csv')
ratings_data = pd.read_csv('ratings.csv')


movie_ratings_merged = pd.merge(ratings_data, movie_names, on= 'movieId')
#print(movie_ratings_merged.head())

avg_rating = movie_ratings_merged.groupby('title')['rating'].mean()
#print(avg_rating)
top_rating = avg_rating.sort_values(ascending=False).head()
print(top_rating)