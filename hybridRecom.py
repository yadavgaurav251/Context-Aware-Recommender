#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import KFold
from surprise.model_selection.validation import cross_validate

print("Import Success")


# In[2]:


# The main Movies Metadata file
meta = pd.read_csv('Dataset/movies_metadata.csv')
meta.head()


# In[3]:


# Rating
ratings = pd.read_csv('Dataset/ratings_small.csv')
ratings.head() # Movies in this dataset are rated out of 5 instead of 10


# In[4]:


#Links of IMDb and TMDb
links = pd.read_csv('Dataset/links_small.csv')
links.head()


# In[5]:


keywords = pd.read_csv('Dataset/keywords.csv')
keywords.head()


# In[6]:


# -- Content-focused recommender --

meta['overview'] = meta['overview'].fillna('')
meta['overview'].head() # Sample descriptions


# In[7]:


# Check the datatype of "id" in movies_metadata.csv
pd.DataFrame({'feature':meta.dtypes.index, 'dtype':meta.dtypes.values})


# In[8]:


meta = meta.drop([19730, 29503, 35587]) # Remove these ids to solve ValueError: "Unable to parse string..."

# Convert object to int64 for compatibility during merging
meta['id'] = pd.to_numeric(meta['id'])

# Run  the following code for converting more than one value to integer
# def convert_int(x):
#     try:
#         return int(x)
#     except:
#         return np.nan


# In[9]:


# Check the datatype of "tmdbId" in links_small.csv
pd.DataFrame({'feature':links.dtypes.index, 'dtype':links.dtypes.values})


# In[10]:


# Convert float64 to int64
col=np.array(links['tmdbId'], np.int64)
links['tmdbId']=col


# In[11]:


# Merge the dataframes on column "tmdbId"
meta.rename(columns={'id':'tmdbId'}, inplace=True)
meta = pd.merge(meta,links,on='tmdbId')
meta.drop(['imdb_id'], axis=1, inplace=True)
meta.head()

# Alternatively, run the following code to reduce the size of movies_metadata.csv to match links_small.csv
# meta = meta[meta['tmdbId'].isin(links)]
# meta.shape


# In[12]:


# Remove stop words and use TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
# Construct TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(meta['overview'])
tfidf_matrix.shape


# In[13]:


# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Get corresponding indices of the movies
indices = pd.Series(meta.index, index=meta['original_title']).drop_duplicates()


# In[14]:


# Recommendation function
def recommend(title, cosine_sim=cosine_sim):

    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 15 most similar movies
    sim_scores = sim_scores[1:16]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Remove low-rated movies or outliers
    for i in movie_indices:
        pop = meta.at[i,'vote_average']
        if pop<5 or pop>10:
            movie_indices.remove(i)

    # Return the most similar movies qualifying the 5.0 rating threshold
    return meta[['original_title','vote_average']].iloc[movie_indices]


# In[15]:


recommend('Iron Man')


# In[16]:


# -- User-focused recommender --

reader = Reader() # Used to parse a file containing ratings
df = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
kf = KFold(n_splits=5)
kf.split(df) # Split the data into folds


# In[17]:


# Use Single Value Decomposition (SVD) for cross-validation and fitting
svd = SVD()
cross_validate(svd, df, measures=['RMSE', 'MAE'])

trainset = df.build_full_trainset()
svd.fit(trainset)


# In[18]:


# Check a random user's ratings
ratings[ratings['userId'] == 10]


# In[19]:


# Read the smaller links file again
links_df = pd.read_csv('Dataset/links_small.csv')
col=np.array(links_df['tmdbId'], np.int64)
links_df['tmdbId']=col

# Merge movies_metadata.csv and links_small.csv files
links_df = links_df.merge(meta[['title', 'tmdbId']], on='tmdbId').set_index('title')
links_index = links_df.set_index('tmdbId') # For label indexing


# In[22]:


# Recommendation function
def hybrid(userId, title):
    idx = indices[title]
    tmdbId = links_df.loc[title]['tmdbId'] # Get the corresponding tmdb id
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31] # Scores of the 30 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    movies = meta.iloc[movie_indices][['title', 'vote_average', 'tmdbId']]
    movies['est'] = movies['tmdbId'].apply(lambda x: svd.predict(userId, links_index.loc[x]['movieId']).est) # Estimated prediction using svd
    movies = movies.sort_values('est', ascending=False) # Rank movies according to the predicted values
    movies.columns = ['Title', 'Vote Average', 'TMDb Id', 'Estimated Prediction']
    return movies.head(15) # Display top 15 similar movies


# In[23]:


hybrid(30,'The Conjuring')


# In[ ]:




