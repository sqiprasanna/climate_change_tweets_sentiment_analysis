#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sqiprasanna/climate_change_tweets_sentiment_analysis/blob/main/code/SVG_code.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt 
from itertools import chain
from wordcloud import WordCloud, STOPWORDS 
# !pip install xlrd==1.2.0
import cufflinks as cf
import regex as re
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
# !pip install xlrd==1.2.0


# In[2]:


get_ipython().run_line_magic('run', './svg_Preprocessing.ipynb')


# In[3]:


# data = pd.read_csv('/content/1377884570_tweet_global_warming.csv',encoding="unicode_escape",engine="python")
data = pd.read_csv('global_warming_tweets.csv',encoding="unicode_escape",engine="python")

data.head()



# In[4]:


print(data['existence'].value_counts())


# In[5]:


data['word_count'] = data['tweet'].apply(lambda x: len(x.split(" ")))
data = data.drop_duplicates()
data = data.dropna()
data.loc[data['existence'] == 'Y','existence'] = "Yes"
data.loc[data['existence'] == 'N','existence'] = "No"
print(data.shape)
data.dropna()
data.loc[data['existence'] == np.nan,'existence'] = "No"
print(data['existence'].value_counts())
print(data.shape)
data.head()


# In[6]:


tweets = data["tweet"]
tweets


# count no of mentions in each tweet

# In[7]:


pattern=r'@\w+'
count_mentions = tweets.apply(lambda tweet: len(re.findall(pattern, tweet)))
count_mentions


# count no of hashtags in each tweet

# In[8]:


pattern=r'#\w+'
print(tweets)
count_hashtags = tweets.apply(lambda tweet: len(re.findall(pattern, tweet)))
print(count_hashtags)


# count no of links in each tweet

# In[9]:


pattern=r'http.?://[^\s]+[\s]?'
count_urls = tweets.apply(lambda tweet: len(re.findall(pattern, tweet)))


# In[97]:


#print the total number of hashtags, mentions and urls.


# In[10]:


print("Mentions:",count_mentions.sum())
print("Hashtags:",count_hashtags.sum())
print("Urls:",count_urls.sum())



# In[11]:


def generateWordCloud(tweets):
    allwords = " ".join(set(chain.from_iterable(tweets)))
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = set(STOPWORDS), 
                    min_font_size = 10).generate(allwords)

    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.figure(figsize = (7, 7), facecolor = 'white', edgecolor='blue') 
    plt.imshow(wordcloud) 

    plt.show()


# In[12]:


preprocessed_tweets,hash_tweets,urls_tweets,mentions_tweets,indices_pre = preprocess(tweets,False)
# preprocessed_tweets


# Checking after prepocessing hashtags and urls 

# In[13]:


pattern='@\w+'
# count_hashtags = preprocessed_tweets.apply(lambda tweet: len(re.findall(pattern, tweet)))
count_hashtags = [len(re.findall(pattern," ".join(t))) for t in preprocessed_tweets]
sum(count_hashtags)


# In[14]:


pattern=r'http.?://[^\s]+[\s]?'
# count_urls = tweets.apply(lambda tweet: len(re.findall(pattern, tweet)))
count_urls = [len(re.findall(pattern," ".join(t))) for t in preprocessed_tweets]
sum(count_urls)


# In[15]:


print(sum(count_hashtags))


# ## VISUALIZATIONS
# https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
# 
# https://github.com/yrtnsari/Sentiment-Analysis-NLP-with-Python/blob/main/wordcloud.ipynb
# 

# In[16]:


pos_tweets=data[data['existence']=='Yes']
pos_tweets.head()


# In[17]:


data_pos=data[data['existence']=='Yes']
hashtags = data_pos["tweet"].apply(lambda x: pd.value_counts(re.findall('(#\w+)', x.lower() ))).sum(axis=0).to_frame().reset_index().sort_values(by=0,ascending=False)
hashtags.columns = ['hashtag','occurences']


# In[18]:


hashtags.head(10)


# In[19]:


data_neg=data[data['existence']=='No']
hashtags = data_neg["tweet"].apply(lambda x: pd.value_counts(re.findall('(#\w+)', x.lower() ))).sum(axis=0).to_frame().reset_index().sort_values(by=0,ascending=False)
hashtags.columns = ['hashtag','occurences']


# In[20]:


hashtags.head(10)


# Collect the positive tweets that contain hashtags

# In[21]:


pre_tweet_pos,pos_hash_tweets,pos_urls_tweets,pos_mentions_tweets,indices_pos=preprocess(pos_tweets.tweet,False)


# In[22]:


generateWordCloud(pre_tweet_pos)


# Negative tweets visualization

# In[23]:


neg_tweets=data[data['existence']=='No']
neg_tweets.head()


# In[24]:


pre_tweet_neg,neg_hash_tweets,neg_urls_tweets,neg_mentions_tweets,indices_neg=preprocess (neg_tweets.tweet,False)


# In[25]:


generateWordCloud(pre_tweet_neg)


# In[125]:


# generateWordCloud([t.split(' ') for t in tweets])
generateWordCloud(preprocessed_tweets)


# In[26]:


import nltk
preprocessed_tweets
flat_list = []
for sublist in preprocessed_tweets:
    for item in sublist:
        flat_list.append(item)


# In[27]:


bigrams_series = (pd.Series(nltk.ngrams(flat_list, 2)).value_counts())[:10]


# In[28]:


bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
plt.title('10 Most Frequently Occuring Bigrams')
plt.ylabel('Bigram')
plt.xlabel('Number of Occurances')


# In[34]:


prepr_tweets = [" ".join(each) for each in preprocessed_tweets]
data_new=data.iloc[indices_pre]
data_new['cleaned_tweet'] = prepr_tweets
data_new.head()


# In[30]:


len(prepr_tweets), data.shape

