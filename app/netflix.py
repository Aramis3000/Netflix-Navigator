#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


netflix_data = pd.read_csv('NetflixDataset.csv',encoding='latin-1', index_col = 'Title')
netflix_data.head(2)


# In[ ]:


netflix_data.index = netflix_data.index.str.title()


# In[ ]:


color = ['blue', 'yellow']
label = ['Series', 'Movies']
sizes = [netflix_data[netflix_data['Series or Movie'] == 'Series'].size, netflix_data[netflix_data['Series or Movie'] == 'Movie'].size]
explode = (0.1, 0)
fig, ax = plt.subplots()
ax.pie(sizes, explode, label, color, '%2.2f%%')
ax.axis('equal')
plt.show()


# In[ ]:


netflix_data.rename(columns={'View Rating':'ViewerRating'}, inplace=True)


# In[ ]:


Language = netflix_data.Languages.str.get_dummies(',')
Lang = Language.columns.str.strip().values.tolist()
Language = netflix_data['Languages']
Language_Count = dict()
for i in Lang:
    p = Language.str.count(i).sum()
    Language_Count[i] = int(p)
print(len(Language_Count))


# In[ ]:


Language_Count = {k: v for k, v in sorted(Language_Count.items(), key=lambda item: item[1], reverse = True)}
top_languages = {"Languages": list(Language_Count.keys()), "Count": list(Language_Count.values())}


# In[ ]:


fig = px.bar(pd.DataFrame(top_languages)[:10], y = 'Languages', x = 'Count', orientation = 'h', title = 'Most Available Languages', color = 'Count', color_continuous_scale = px.colors.qualitative.Prism).update_yaxes(categoryorder = 'total ascending')
fig.show()


# In[ ]:


Genres = netflix_data.Genre.str.get_dummies(',')
Genre = Genres.columns.str.strip().values.tolist()
Genres = netflix_data['Genre']
Genre_Count = dict()
for i in Genre:
    p = Genres.str.count(i).sum()
    Genre_Count[i] = int(p)
print(len(Genre_Count))


# In[ ]:


Genre_Count = {k: v for k, v in sorted(Genre_Count.items(), key=lambda item: item[1], reverse = True)}
top_genres = {"Genre": list(Genre_Count.keys()), "Count": list(Genre_Count.values())}


# In[ ]:


fig = px.bar(pd.DataFrame(top_genres)[:10], y = 'Genre', x = 'Count', orientation = 'h', title = 'Genres with maximum content', color = 'Count', color_continuous_scale = px.colors.qualitative.Prism).update_yaxes(categoryorder = 'total ascending')
fig.show()


# In[ ]:


top_15 = netflix_data.sort_values(by = ['IMDb Score'], ascending = False).head(15)
plt.figure(figsize = (15,5))
sns.barplot(data = top_15, y = top_15.index, x = "IMDb Score")
plt.show()


# In[ ]:


netflix_data = netflix_data[~netflix_data.index.duplicated()]


# In[ ]:


netflix_data.index.duplicated().sum()


# In[ ]:


netflix_data.index.isnull().sum()


# In[ ]:


netflix_data['Genre'] = netflix_data['Genre'].astype('str')
print((netflix_data['Genre'] == 'nan').sum())


# In[ ]:


netflix_data['Tags'] = netflix_data['Tags'].astype('str')
print((netflix_data['Tags'] == 'nan').sum())


# In[ ]:


print(((netflix_data['Genre'] == 'nan') & (netflix_data['Tags'] == 'nan')).sum())
#so these two features can used to recommend movies as no movie can be left unrecommended


# In[ ]:


print(netflix_data[['IMDb Score']].describe())
netflix_data['IMDb Score'].mode()
#this feature will be used to sort the movie or series list to represent the recommended items


# In[ ]:


netflix_data['IMDb Score'] = netflix_data['IMDb Score'].apply(lambda x: 6.6 if x == 0 or math.isnan(x) else x)
print(netflix_data[['IMDb Score']].describe())
#since no value has suffered for change greater than 0.0003 after replacing the null values with mode value, so we replace the null values with 6.6


# In[ ]:


netflix_data['Actors'] = netflix_data['Actors'].astype('str')
netflix_data['ViewerRating'] = netflix_data['ViewerRating'].astype('str')


# In[ ]:


def prepare_data(x):
        return str.lower(x.replace(" ", ""))


# In[ ]:


new_features = ['Genre', 'Tags', 'Actors', 'ViewerRating']
selected_data = netflix_data[new_features]


# In[ ]:


for new_feature in new_features:
    selected_data.loc[:, new_feature] = selected_data.loc[:, new_feature].apply(prepare_data)
selected_data.index = selected_data.index.str.lower()
selected_data.index = selected_data.index.str.replace(" ",'')
selected_data.head(2)


# In[ ]:


def create_soup(x):
    return x['Genre'] + ' ' + x['Tags'] + ' ' +x['Actors']+' '+ x['ViewerRating']


# In[ ]:


selected_data.loc[:, 'soup'] = selected_data.apply(create_soup, axis = 1)
selected_data.head(2)


# In[ ]:


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(selected_data['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[ ]:


selected_data.reset_index(inplace = True)
selected_data.head(2)


# In[ ]:


indices = pd.Series(selected_data.index, index=selected_data['Title'])
indices


# In[ ]:


result = 0
def get_recommendations(title, cosine_sim):
    global result
    title=title.replace(' ','').lower()
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 50 most similar movies
    sim_scores = sim_scores[1:51]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    result =  netflix_data.iloc[movie_indices]
    result.reset_index(inplace = True)
    return result


# In[ ]:


df = pd.DataFrame()
movienames = ['Annabelle Comes Home','The Nun', 'Insidious: The Last Key', 'Conjuring 2', 'Insidious: Chapter 3']
languages = ['English', 'Hindi']
for moviename in movienames:
    get_recommendations(moviename,cosine_sim2)
    for language in languages:
        df = pd.concat([result[result['Languages'].str.count(language) > 0], df], ignore_index=True)
df.drop_duplicates(keep = 'first', inplace = True)
df.sort_values(by = 'IMDb Score', ascending = False, inplace = True)


# In[ ]:


print(df.shape)
print(df.head())

