import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#Loading the dataset
data=pd.read_csv("movies.csv")
selected_features=['genres','keywords','tagline','cast','director']
# replacing the null values with the null string
for i in selected_features:
    data[i]=data[i].fillna(' ')
# combining all the features
combined_features=data['genres']+' '+data['keywords']+' '+data['tagline']+' '+data['cast']+' '+data['director']
#Converting the data into feature vectors
vectorizer=TfidfVectorizer()
feature_vector=vectorizer.fit_transform(combined_features)
#getting the similarity constant score
similarity=cosine_similarity(feature_vector)
mob=[]

def recomended_movies(title):
    movie_name=title
    movies=[]
    #Creating the list of movies in the dataset
    list_movies=data['title'].tolist()
    #finding the close match for the movie name given by the user.
    find_close_match=difflib.get_close_matches(movie_name,list_movies) 
    close_match=find_close_match[0]
    title_index=data[data.title==close_match]['index'].values[0]
    #getting a  list of similar movies
    similarity_score=list(enumerate(similarity[title_index]))
    sorted_similar_movies=sorted(similarity_score, key=lambda x:x[1], reverse=True)
    i=1
    
    for movie in sorted_similar_movies:
        index=movie[0]
        title_from_index=data[data.index==index]['title'].values[0]
        if(i<10):
            #print(i,' .',title_from_index)
            movies.append(title_from_index)
            i+=1
    mob=movies
    return mob
    
 
 