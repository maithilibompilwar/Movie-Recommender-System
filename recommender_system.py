import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
print(movies)
print(credits)
movies.shape
credits.shape
credits.head(2)
movies.head(2)
movies = movies.merge(credits, on = "title")
movies.shape
movies = movies[["id", 'title', 'genres', 'overview','keywords','cast','crew']]
movies.shape
movies.head(2)
movies.isnull().sum()
movies.dropna(inplace=True) #delete permanently
movies.isnull().sum()
movies.shape
movies.duplicated().sum()
movies.head(2)
movies[['genres']]
movies.iloc[0]['genres']
import ast #convert string to list

def convert(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l
    
movies['genres'] = movies['genres'].apply(convert)
movies['genres']
movies.head(2)
movies.iloc[0]['keywords']
movies['keywords']=movies['keywords'].apply(convert)
movies['keywords']
movies.head(2)
movies.iloc[0]['cast']
def convert_cast(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            l.append(i['name'])
        counter += 1
    return l
movies['cast']=movies['cast'].apply(convert_cast)
movies.head(2)
movies.iloc[0]['crew']
def fetch_director(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            
            l.append(i['name'])
            break
      
    return l
movies['crew']=movies['crew'].apply(fetch_director)
movies.head(2)
movies.iloc[0]['overview']#converting the overview string type to list, to concatenate easily
movies['overview']=movies['overview'].apply(lambda x:x.split())
movies.head(2)
movies.iloc[0]['overview']
#convert Sam Worthington >>>>> SamWorthington
def remove_space(word):
    l = []
    for i in word:
        l.append(i.replace(" ",""))
    return l
movies['cast']=movies['cast'].apply(remove_space)
movies['crew']=movies['crew'].apply(remove_space)
movies['genres']=movies['genres'].apply(remove_space)
movies['keywords']=movies['keywords'].apply(remove_space)
movies.head(2)
#generating tags
movies['tags'] = movies['overview']+ movies['genres']+ movies['keywords']+ movies['cast']+ movies['crew']
movies.head(2)
movies.iloc[0]['tags']
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words= 'english')
vector = cv.fit_transform(new_df['tags']).toarray()
vector
vector.shape
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
similarity
similarity.shape
new_df[new_df['title'] == 'Spider-Man'].index[0]
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
recommend('The Dark Knight Rises')
