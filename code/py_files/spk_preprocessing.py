#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt 
from itertools import chain
from wordcloud import WordCloud, STOPWORDS 
from sklearn.feature_extraction.text import CountVectorizer

# !pip install xlrd==1.2.0
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
# !pip install xlrd==1.2.0


# In[2]:


# ### Source: https://spacy.io/usage/linguistic-features

# def spacyPipeline(tweets):
#     ps = PorterStemmer()
#     nlp = spacy.load('en_core_web_sm')
    
#     preprocessed_tweets = []
#     for t in tweets:
#         doc = nlp(t)
#         filtered_tweet = []
        
#         for token in doc:
#             if (not token.is_stop) and token.is_alpha:
#                 filtered_tweet.append(ps.stem(str(token)))
        
#         preprocessed_tweets.append(filtered_tweet)
    
#     return preprocessed_tweets


# In[3]:


app_words = ['bit', 'ly']
STOPWORDS.update(app_words)


# In[4]:


def generateWordCloud(tweets):
    allwords = " ".join(set(chain.from_iterable(tweets)))
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = set(STOPWORDS), 
                    min_font_size = 10).generate(allwords)

    plt.axis("off") 
#     plt.tight_layout(pad = 0) 

    plt.figure(figsize = (7, 7), facecolor = 'white', edgecolor='blue') 
    plt.imshow(wordcloud) 

    plt.show()


# In[5]:


def preprocess_tweets(tweets):
    # Convert all to lowercase
    tweets = [t.lower() for t in tweets]
    
    # Process tweets through spaCy pipeline
    tweets,indices = spacyPipeline(tweets)
    
    # Filter out words
    tweets = [list(filter(lambda w: w != 'link', t)) for t in tweets]

    # Remove words less than length 2
#     tweets = [list(filter(lambda w: len(w) > 2, t)) for t in tweets]
    
#     print(tweets)
    return tweets, indices


# In[6]:


def get_top_n_words(corpus,min_n,max_n, n=None):
    vec = CountVectorizer(stop_words=STOPWORDS, ngram_range=(min_n,max_n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if len(word) > 3 and word not in set(STOPWORDS)]
    # words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[7]:


def plot_conf_matrix(y_test,predicted):
    f, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(confusion_matrix(y_test, predicted), annot=True, fmt=".0f", ax=ax)
    plt.xlabel("y_head")
    plt.ylabel("y_true")
    plt.show()


# In[ ]:




