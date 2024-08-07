import numpy as np
import pandas as pd
from math import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def popular_based():
#popularity based recommendation
    movie_names = pd.read_csv('movies.csv')
    ratings_data = pd.read_csv('ratings.csv')


    movie_ratings_merged = pd.merge(ratings_data, movie_names, on= 'movieId')
    #print(movie_ratings_merged.head())

    avg_rating = movie_ratings_merged.groupby('title')['rating'].mean()
    #print(avg_rating)
    #top_rating = avg_rating.sort_values(ascending=False)
    #print(top_rating)


    most_views = movie_ratings_merged.groupby('title')['rating'].count()
    #print(most_views)
    movie_most = pd.merge(avg_rating, most_views, on= 'title')
    movie_most.columns = ['rating', 'view_count']

    movie_most=movie_most.sort_values(by='rating', ascending=False)
    movie_most = movie_most[(movie_most['rating'] > 4) & (movie_most['view_count'] >200 )]
    return movie_most[:10]

def content_based(title):
#content based recommender system (does not need data from other users)
#done with cosine eximilarity
    df_movies = pd.read_csv('tmdb_5000_movies.csv')
    df_credits = pd.read_csv('tmdb_5000_credits.csv')
    #print(df_movies.columns)
    tf_idf = TfidfVectorizer(stop_words='english')
    df_movies['overview']= df_movies['overview'].fillna("") #replaces all nans with empty string so doesnt break vectorizer
    tf_idf_matrix = tf_idf.fit_transform(df_movies['overview'])
    consine_sim = linear_kernel(tf_idf_matrix, tf_idf_matrix)
    index = pd.Series(df_movies.index, index=df_movies['original_title']).drop_duplicates()
    
    #given movie title, get the index, then correspond with similarity
    idx = index[title]
    similarity_score = enumerate(consine_sim[idx]) #creates tuples with index, score
    similarity_score = sorted(similarity_score, key = lambda x: x[1], reverse = True)
    print(similarity_score[1:6])
    top_close_movies = []
    for x in similarity_score[1:6]:
        top_close_movies.append(df_movies['original_title'][x[0]])
    print(top_close_movies)
    return top_close_movies


#print(popular_based())
content_based('The Dark Knight Rises')