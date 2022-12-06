#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt 
from itertools import chain
from wordcloud import WordCloud, STOPWORDS 
from nltk.stem import PorterStemmer
import re

import cufflinks as cf
# cf.go_offline()
# cf.set_config_file(offline=False, world_readable=True)

get_ipython().run_line_magic('run', './jlu_preprocessing.ipynb')
get_ipython().run_line_magic('run', './spk_preprocessing.ipynb')


# In[2]:


# from google.colab import drive
# drive.mount('/content/gdrive')


# In[3]:


# data = pd.read_excel(r'C:/Users/Checkout/Desktop/SJSU/sem1/257-ML/Project/global_warming_tweets.xls')
data = pd.read_csv(r'C:/Users/Checkout/Desktop/SJSU/sem1/257-ML/Project/global_warming_tweets_main.csv', engine='python') #encoding = "cp1252"
data.head()


# In[4]:


data[data['existence']=="No"].tweet


# In[5]:


print(data['existence'].value_counts())


# In[6]:


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


# In[7]:


data['tweet']


# In[8]:


tweets = data["tweet"]
# tweets = tweets.drop_duplicates()
tweets


# In[9]:


# tweets[:100]


# In[10]:


# python -m spacy download en_core_web_sm
preprocessed_tweets,indices = preprocess_tweets(tweets)


# In[11]:


# preprocessed_tweets


# ## VISUALIZATIONS
# https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
# 
# https://github.com/yrtnsari/Sentiment-Analysis-NLP-with-Python/blob/main/wordcloud.ipynb
# 

# In[22]:


# generateWordCloud([t.split(' ') for t in tweets])
pos_pre_tweets,pos_pre_indices = preprocess_tweets(data[data['existence'] == "Yes"].tweet)
neg_pre_tweets,neg_pre_indices = preprocess_tweets(data[data['existence'] == "No"].tweet)
generateWordCloud(pos_pre_tweets)
generateWordCloud(neg_pre_tweets)


# In[23]:


data['existence.confidence'].plot(
    kind='hist',
    bins=30,
    title='Tweet Sentiment Distribution ')


# In[31]:


data[data.existence == "Yes"]['word_count'].plot(
    kind='hist',
    xlabel = "Word Count",
    bins=50,
    title='Tweet Word Count Distribution')
data[data.existence == "No"]['word_count'].plot(
    kind='hist',
    xlabel = "Word Count",
    bins=50,
    title='Tweet Word Count Distribution')


# In[25]:


print(data.groupby(by=["existence"]).sum())
data.groupby(by=["existence"]).sum().plot(kind = 'bar',title = "Existence Distribution")


# ## Top Uni-gram Words 

# In[26]:


# The distribution of top unigrams before removing stop words
from sklearn.feature_extraction.text import CountVectorizer
min_n, max_n = 1,1
common_words = get_top_n_words(data['tweet'], min_n, max_n,20)
print("Top 20 words:-")
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['tweet' , 'word_count'])
display(df1.head())

df1.groupby('tweet').sum()['word_count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words in review before preprocessing')


# In[27]:


min_n, max_n = 1,1
prepr_tweets = [" ".join(each) for each in preprocessed_tweets]

common_words = get_top_n_words(prepr_tweets, min_n, max_n,20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['tweet' , 'word_count'])
display(df1.head())

df1.groupby('tweet').sum()['word_count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words in review before preprocessing')

# allwords = list(chain.from_iterable(preprocessed_tweets))
# df2 = pd.DataFrame(allwords,columns = ["words"])
# df2['words'].value_counts()[:20].plot(kind='bar', title='Top 20 words in review after preprocessing',
#                                      xlabel = 'top words')


# ## Top Bi-gram Words 

# In[28]:


# The distribution of top unigrams before removing stop words
min_n, max_n = 2,2
common_words = get_top_n_words(data['tweet'], min_n, max_n,20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['tweet' , 'word_count'])
display(df1.head())

df1.groupby('tweet').sum()['word_count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words in review before preprocessing')


# In[29]:


min_n, max_n = 2,2
prepr_tweets = [" ".join(each) for each in preprocessed_tweets]

common_words = get_top_n_words(prepr_tweets, min_n, max_n,20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['tweet' , 'word_count'])
display(df1.head())

df1.groupby('tweet').sum()['word_count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words in review before preprocessing')

# allwords = list(chain.from_iterable(preprocessed_tweets))
# df2 = pd.DataFrame(allwords,columns = ["words"])
# df2['words'].value_counts()[:20].plot(kind='bar', title='Top 20 words in review after preprocessing',
#                                      xlabel = 'top words')


# In[ ]:





# In[ ]:





# In[20]:


new_data = data.iloc[indices]


# In[21]:


prepr_tweets = [" ".join(each) for each in preprocessed_tweets]
new_data['cleaned_tweet'] = prepr_tweets
new_data.head()


# In[ ]:


len(prepr_tweets), data.shape, new_data.shape

