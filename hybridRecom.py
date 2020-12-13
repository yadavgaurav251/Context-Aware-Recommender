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
import copy
from datetime import datetime


print("Import Success")


# In[2]:


meta = pd.read_csv('Dataset/movies_metadata.csv')
meta.head()


# In[3]:


# Rating
ratings = pd.read_csv('Dataset/ratings_small.csv')
ratings.head()


# In[4]:


#Links of IMDb and TMDb
links = pd.read_csv('Dataset/links_small.csv')
links.head()


# In[5]:


keywords = pd.read_csv('Dataset/keywords.csv')
keywords.head()


# In[6]:


# Content based Recommender System

meta['overview'] = meta['overview'].fillna('')
meta['overview'].head()


# In[7]:


pd.DataFrame({'feature':meta.dtypes.index, 'dtype':meta.dtypes.values})


# In[8]:


meta = meta.drop([19730, 29503, 35587]) # Remove these ids to solve ValueError: "Unable to parse string..."

meta['id'] = pd.to_numeric(meta['id'])


# In[9]:


pd.DataFrame({'feature':links.dtypes.index, 'dtype':links.dtypes.values})


# In[10]:


col=np.array(links['tmdbId'], np.int64)
links['tmdbId']=col


# In[11]:


meta.rename(columns={'id':'tmdbId'}, inplace=True)
meta = pd.merge(meta,links,on='tmdbId')
meta.drop(['imdb_id'], axis=1, inplace=True)
meta.head()


# In[12]:


tfidf = TfidfVectorizer(stop_words='english')


# Constructing  matrix TF-IDF
tfidf_matrix = tfidf.fit_transform(meta['overview'])
tfidf_matrix.shape


# In[13]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(meta.index, index=meta['original_title']).drop_duplicates()


# In[14]:


def recommend(title, cosine_sim=cosine_sim):

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:16]
    movie_indices = [i[0] for i in sim_scores]
    for i in movie_indices:
        pop = meta.at[i,'vote_average']
        if pop<5 or pop>10:
            movie_indices.remove(i)

    return meta[['original_title','vote_average']].iloc[movie_indices]


# In[15]:


recommend('Iron Man')


# In[16]:



reader = Reader() 
df = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
kf = KFold(n_splits=5)
kf.split(df) 


# In[ ]:


svd = SVD()
cross_validate(svd, df, measures=['RMSE', 'MAE'])

trainset = df.build_full_trainset()
svd.fit(trainset)


# In[ ]:


ratings[ratings['userId'] == 10]


# In[ ]:


# smaller link file reload 
links_df = pd.read_csv('Dataset/links_small.csv')
col=np.array(links_df['tmdbId'], np.int64)
links_df['tmdbId']=col

links_df = links_df.merge(meta[['title', 'tmdbId']], on='tmdbId').set_index('title')
links_index = links_df.set_index('tmdbId') 


# In[ ]:


def hybrid(userId, title):
    idx = indices[title]
    tmdbId = links_df.loc[title]['tmdbId'] # Get the corresponding tmdb id
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31] # Scores of 30 similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    movies = meta.iloc[movie_indices][['title', 'vote_average', 'tmdbId']]
    movies['est'] = movies['tmdbId'].apply(lambda x: svd.predict(userId, links_index.loc[x]['movieId']).est) # Estimated prediction using svd
    movies = movies.sort_values('est', ascending=False) # Rank movies according to the predicted values
    movies.columns = ['Title', 'Vote Average', 'TMDb Id', 'Estimated Prediction']
    return movies.head(30) # Display top 30 recommended movies


# In[ ]:


hybrid(30,'The Conjuring')

result = hybrid(30,'Batman Begins')
print("data getting passed in contextual")
print(result)


# In[ ]:


# necessary functions for contextual_update function

def day_time():
    now = datetime.now().time()
    morning=now.replace(hour=12,minute=0,second=0,microsecond=0)
    afternoon=now.replace(hour=16,minute=0,second=0,microsecond=0)
    evening=now.replace(hour=19,minute=0,second=0,microsecond=0)

    if now< morning :
        return "morning"
    elif now<afternoon :
        return "afternoon"
    elif now<evening :
        return "evening"
    else :
        return "night"

def season():
    month = datetime.now().month

    if month < 4:
        return "winter"
    elif month <6:
        return "summer"
    elif month <9:
        return "rainy"
    elif month < 11:
        return "autumn"
    else :
        return  "winter"

def is_weekend():
    day=datetime.now().isoweekday()
    if day< 6:
        return False
    return True


#testing function
#day_time()
season()


# In[ ]:


# Function to include movies on specific dates -
def special_date(recommended_list,date_passed):
    print("special date function reached")
    date_event = datetime.now().date()

    # Independence Day
    date_event=date_event.replace(month=8,day=15)
    new_list=recommended_list.copy()
    if date_event == date_passed:
        # Vote Average  TMDb Id  Estimated Prediction
        new_movie = pd.DataFrame({"Title":["Border","Uri:The Surgical Strike"],
                                  "Vote Average":[6.8,7.1],
                                  "TMDb Id":[33125,554600],
                                  "Estimated Prediction":[5.0,5.0],
                                  "tmdbId":[33125,554600],
                                  "genres":["[{'name':'Action'},{'name':'History'},{'name':'War'}]","[{'name':'Action'},{'name':'Drama'},{'name':'War'}]"]
                                  })
        new_list = pd.concat([new_movie,recommended_list])
    #Repubic Day
    date_event=date_event.replace(month=1,day=26)
    if date_event == date_passed:
        new_movie = pd.DataFrame({"Title":["Shaheed","Border","Uri:The Surgical Strike"],
                                  "Vote Average":[5.0,6.8,7.1],
                                  "TMDb Id":[498713,33125,554600],
                                  "Estimated Prediction":[5.0,5.0,5.0],
                                  "tmdbId":[498713,33125,554600],
                                  "genres":["[{'name':'War'},{'name':'History'}]","[{'name':'Action'},{'name':'History'},{'name:'War'}]","[{'name':'Action'},{'name':'Drama'},{'name':'War'}]"]
                                  })
        new_list = pd.concat([new_movie,recommended_list])
    #Teachers Day
    date_event=date_event.replace(month=9,day=5)
    if date_event == date_passed:
        new_movie = pd.DataFrame({"Title":["Super 30","Taare Zameen Par"],
                                  "Vote Average":[7.6,8.0],
                                  "TMDb Id":[534075,7508],
                                  "Estimated Prediction":[5.0,5.0],
                                  "tmdbId":[534075,7508],
                                  "genres":["[{'name':'Drama'}]","[{'name':'Drama'}]"]
                                  })
        new_list = pd.concat([new_movie,recommended_list])
    #Children day
    date_event=date_event.replace(month=11,day=14)
    if date_event == date_passed:
        new_movie = pd.DataFrame({"Title":["Taare Zameen Par","Chillar Party"],
                                  "Vote Average":[8.0,6.9],
                                  "TMDb Id":[7508,69891],
                                  "Estimated Prediction":[5.0,5.0],
                                  "tmdbId":[7508,69891],
                                  "genres":["[{'name':'Drama'}]","[{'name':'Drama'},{'name':'Comedy'},{'name':'Family'}]"]
                                  })
        new_list = pd.concat([new_movie,recommended_list])
    #Christmas
    date_event=date_event.replace(month=12,day=25)
    if date_event == date_passed:
        new_movie = pd.DataFrame({"Title":["Let It Snow","Home Alone"],
                                  "Vote Average":[6.1,7.3],
                                  "TMDb Id":[295151,771],
                                  "Estimated Prediction":[5.0,5.0],
                                  "tmdbId":[295151,771],
                                  "genres":["[{'name':'Romance'},{'name':'Comedy'}]","[{'name':'Comedy'},{'name':'Family'}]"]
                                  })
        new_list = pd.concat([new_movie,recommended_list])
    #New Year
    date_event=date_event.replace(month=12,day=31)
    if date_event == date_passed:
        new_movie = pd.DataFrame({"Title":["New Years Eve"],
                                  "Vote Average":[5.9],
                                  "TMDb Id":[62838],
                                  "Estimated Prediction":[5.0],
                                  "tmdbId":[62838],
                                  "genres":["[{'name':'Comedy'},{'name':'Romance'}]"]
                                  })
        new_list = pd.concat([new_movie,recommended_list])
    date_event=date_event.replace(month=1,day=1)
    if date_event == date_passed:
        new_movie = pd.DataFrame({"Title":["New Years Eve"],
                                  "Vote Average":[5.9],
                                  "TMDb Id":[62838],
                                  "Estimated Prediction":[5.0],
                                  "tmdbId":[62838],
                                  "genres":["[{'name':'Comedy'},{'name':'Romance'}]"]
                                  })
        new_list = pd.concat([new_movie,recommended_list])
    #Valentine
    date_event=date_event.replace(month=2,day=14)
    if date_event == date_passed:
        new_movie = pd.DataFrame({"Title":["The Notebook","Titanic"],
                                  "Vote Average":[7.9,7.9],
                                  "TMDb Id":[11036,597],
                                  "Estimated Prediction":[5.0,5.0],
                                  "tmdbId":[11036,597],
                                  "genres":["[{'name':'Romance'},{'name':'Drama'}]","[{'name':'Drama'},{'name':'Romance'}]"]
                                  })
        new_list = pd.concat([new_movie,recommended_list])
    
    return new_list


# In[ ]:


def recommendation_updater(recommended_list,genre_score):
    #print("reached recommendation updater - ")
    new_list=recommended_list.copy()
    for ind in recommended_list.index:
        new_score=0
        movie_genre= list(eval(recommended_list['genres'][ind]))
        #print(recommended_list['genres'][ind])
        #print(type(recommended_list['genres'][ind]))
        #print(movie_genre)
        curr_genre_list= [li['name'] for li in movie_genre]
        #print(curr_genre_list)
        for genre in curr_genre_list:
            if genre in genre_score:
                new_score+=genre_score[genre]
        #print(new_score)
        new_list['Estimated Prediction'][ind]=new_list['Estimated Prediction'][ind]+new_score
    return new_list


# In[ ]:



def contextual_update(list_passed,family=False,device="Mobile",no_of_people=1,date_passed=datetime.now().date()) :
    # categories we have romance,action,comedy,drama ,crime and thriller ,documentary,sci-fi
    recommended_list=list_passed.copy()
    print("Before Context-Awareness based changes - ")
    print(list_passed)

    # Adding Genres for update
    recommended_list = pd.merge(recommended_list,meta[['tmdbId','genres']],left_on=['TMDb Id'],right_on=['tmdbId']).dropna()

    # Special Days
    test_date=datetime.now().date()
    test_date=test_date.replace(month=8,day=15)
    recommended_list=special_date(recommended_list,test_date)
    recommended_list.reset_index(drop=True,inplace=True)

    # Reducing score to take account for contextual_update

    effect_rate = 0.75
    category=4
    recommended_list['Estimated Prediction']=recommended_list['Estimated Prediction']-effect_rate


    # Timing based

    day_part = day_time()
    if day_part == "morning":
        scores={
        'Romance':0.24*(effect_rate/category),'Action':0.18*(effect_rate/category),'Comedy':0.64*(effect_rate/category),'Drama':0.24*(effect_rate/category),'Crime':0.17*(effect_rate/category)
        ,'Thriller':0.17*(effect_rate/category),'Documentary':0.25*(effect_rate/category),'Science Fiction':0.28*(effect_rate/category)
        }
    elif day_part =="afternoon":
        scores ={
           'Romance':0.18*(effect_rate/category),'Action':0.44*(effect_rate/category),'Comedy':0.48*(effect_rate/category),'Drama':0.35*(effect_rate/category),'Crime':0.5*(effect_rate/category)
        ,'Thriller':0.5*(effect_rate/category),'Documentary':0.24*(effect_rate/category),'Science Fiction':0.35*(effect_rate/category)
        }
    elif day_part =="evening":
        scores={
        'Romance':0.4*(effect_rate/category),'Action':0.34*(effect_rate/category),'Comedy':0.48*(effect_rate/category),'Drama':0.3*(effect_rate/category),'Crime':0.4*(effect_rate/category)
        ,'Thriller':0.4*(effect_rate/category),'Documentary':0.24*(effect_rate/category),'Science Fiction':0.32*(effect_rate/category)
        }
    else :
        scores={
        'Romance':0.57*(effect_rate/category),'Action':0.37*(effect_rate/category),'Comedy':0.42*(effect_rate/category),'Drama':0.37*(effect_rate/category),'Crime':0.54*(effect_rate/category)
        ,'Thriller':0.54*(effect_rate/category),'Documentary':0.31*(effect_rate/category),'Science Fiction':0.41*(effect_rate/category)
        }
    recommended_list=recommendation_updater(recommended_list,scores)

    # Season based
    curr_season = season()
    if curr_season == "summer":
        scores={
        'Romance':0.32*(effect_rate/category),'Action':0.48*(effect_rate/category),'Comedy':0.57*(effect_rate/category),'Drama':0.5*(effect_rate/category),'Crime':0.6*(effect_rate/category)
        ,'Thriller':0.6*(effect_rate/category),'Documentary':0.27*(effect_rate/category),'Science Fiction':0.47*(effect_rate/category)
        }
    elif curr_season == "rainy":
        scores={
        'Romance':0.57*(effect_rate/category),'Action':0.3*(effect_rate/category),'Comedy':0.52*(effect_rate/category),'Drama':0.5*(effect_rate/category),'Crime':0.41*(effect_rate/category)
        ,'Thriller':0.41*(effect_rate/category),'Documentary':0.14*(effect_rate/category),'Science Fiction':0.32*(effect_rate/category)
        }
    elif curr_season == "autumn":
        scores={
        'Romance':0.41*(effect_rate/category),'Action':0.37*(effect_rate/category),'Comedy':0.5*(effect_rate/category),'Drama':0.48*(effect_rate/category),'Crime':0.52*(effect_rate/category)
        ,'Thriller':0.52*(effect_rate/category),'Documentary':0.31*(effect_rate/category),'Science Fiction':0.44*(effect_rate/category)
        }
    else :
        scores={
        'Romance':0.54*(effect_rate/category),'Action':0.45*(effect_rate/category),'Comedy':0.51*(effect_rate/category),'Drama':0.42*(effect_rate/category),'Crime':0.5*(effect_rate/category)
        ,'Thriller':0.5*(effect_rate/category),'Documentary':0.21*(effect_rate/category),'Science Fiction':0.32*(effect_rate/category)
        }
    recommended_list=recommendation_updater(recommended_list,scores)

    # Weekday based -

    if is_weekend():
        scores={
        'Romance':0.41*(effect_rate/category),'Action':0.48*(effect_rate/category),'Comedy':0.54*(effect_rate/category),'Drama':0.38*(effect_rate/category),'Crime':0.7*(effect_rate/category)
        ,'Thriller':0.7*(effect_rate/category),'Documentary':0.28*(effect_rate/category),'Science Fiction':0.41*(effect_rate/category)
        }
    else :
        scores={
        'Romance':0.37*(effect_rate/category),'Action':0.32*(effect_rate/category),'Comedy':0.51*(effect_rate/category),'Drama':0.32*(effect_rate/category),'Crime':0.48*(effect_rate/category)
        ,'Thriller':0.48*(effect_rate/category),'Documentary':0.21*(effect_rate/category),'Science Fiction':0.38*(effect_rate/category)
        }

    recommended_list=recommendation_updater(recommended_list,scores)

    # Device Based

    if  device == "phone":
        scores={
        'Romance':0.36*(effect_rate/category),'Action':0.24*(effect_rate/category),'Comedy':0.66*(effect_rate/category),'Drama':0.44*(effect_rate/category),'Crime':0.38*(effect_rate/category)
        ,'Thriller':0.38*(effect_rate/category),'Documentary':0.2*(effect_rate/category),'Science Fiction':0.21*(effect_rate/category)
        }
    elif device =="tablet":
        scores={
        'Romance':0.34*(effect_rate/category),'Action':0.37*(effect_rate/category),'Comedy':0.43*(effect_rate/category),'Drama':0.43*(effect_rate/category),'Crime':0.42*(effect_rate/category)
        ,'Thriller':0.42*(effect_rate/category),'Documentary':0.22*(effect_rate/category),'Science Fiction':0.36*(effect_rate/category)
        }
    else :
        scores={
        'Romance':0.33*(effect_rate/category),'Action':0.6*(effect_rate/category),'Comedy':0.24*(effect_rate/category),'Drama':0.3*(effect_rate/category),'Crime':0.66*(effect_rate/category)
        ,'Thriller':0.66*(effect_rate/category),'Documentary':0.21*(effect_rate/category),'Science Fiction':0.58*(effect_rate/category)
        }
    recommended_list=recommendation_updater(recommended_list,scores)

    # Based on Number of people and Family -

    if no_of_people >1 :
        if family:
            scores={
            'Romance':0.1*(effect_rate/category),'Action':0.43*(effect_rate/category),'Comedy':0.66*(effect_rate/category),'Drama':0.49*(effect_rate/category),'Crime':0.26*(effect_rate/category)
        ,'Thriller':0.26*(effect_rate/category),'Documentary':0.36*(effect_rate/category),'Science Fiction':0.29*(effect_rate/category)
        }
        else :
            scores={
        'Romance':0.33*(effect_rate/category),'Action':0.63*(effect_rate/category),'Comedy':0.54*(effect_rate/category),'Drama':0.33*(effect_rate/category),'Crime':0.61*(effect_rate/category)
        ,'Thriller':0.61*(effect_rate/category),'Documentary':0.17*(effect_rate/category),'Science Fiction':0.54*(effect_rate/category)
        }

        recommended_list=recommendation_updater(recommended_list,scores)

    # removing genre from table

    recommended_list.drop(['tmdbId','genres'],axis=1,inplace=True)

    # Sorting the list for final result and comparing
    #print(list_passed)
    recommended_list.sort_values(by='Estimated Prediction',ascending=False,inplace=True)
    print(recommended_list)

contextual_update(result)


# In[ ]:



# In[ ]:




