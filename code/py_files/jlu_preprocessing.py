#!/usr/bin/env python
# coding: utf-8

# In[3]:


import re
import numpy as np
import pandas as pd
import spacy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import enchant


# In[4]:


def splitWords(t):
    splitTweet = []
    t = t.split(' ')
    
    for word in t:
        # Not splitting links
        if re.search(r'(bit.ly)|(http)|(.com)', word) is not None:
            splitTweet.append(word)
            continue
        
        # Split words conjoined together by punctuation i.e. "act|Brussels"
        words = re.split(r'["|,;!|#:*&]', word)
        
        # Split words that are conjoined together i.e. "EarthDay"
        for word in words:
            # Remove words that are too short or contain special characters
            if len(word) < 3 or re.search(r'[À-ȕ]', word) is not None:
                continue
            
            res = re.search(r'[A-Z]{1}[a-z]{1,}[A-Z]{1}', word)

            if res is not None:
                i = res.span()[1]
                splitTweet.extend([word[:i-1], word[i-1:]])
            else:
                splitTweet.extend([word])
    
    return ' '.join(splitTweet)


# In[5]:


def checkToken(token):
    fillers = set({'link', 'http'})
    common = set() #set({'global', 'warming', 'climate', 'change'}) # Remove common signal between tweets
    return (len(token.text) > 3) and (not token.is_stop) and (token.is_alpha) and (token.text.lower() not in fillers) and (token.text.lower() not in common)


# In[6]:


def spacyPipeline(tweets, verbose=False):
    nlp = spacy.load('en_core_web_sm')
    enchant_dict = enchant.Dict("en_US")
    
    MIN_TWEET_LEN = 4
    
    indices = []
    preprocessed_tweets = []
    for index, t in enumerate(tweets):
           
        if verbose:
            print(t)
        
        # Tokenizing tweet with spaCy
        doc = nlp(t)
        filtered_tweet = set()
        
        # Finding country or city names
        locs = set()
        for ent in doc.ents:
            if ent.label_ == "GPE" or ent.label_ == "LOC":
                for word in ent.text.lower().split(' '):
                    locs.add(word)
                        
        if verbose:
            print("locs: ", locs)
        
        # Filter through words 
        for token in doc:
            # Check for words not in english dictionary
            if not enchant_dict.check((str(token.text.lower()))): 
                continue
                
            # Check for duplicate words
            if token.lemma_ in filtered_tweet:
                continue
                
            # print(token, " | ", spacy.explain(token.pos_))
            
            if (token.text.lower() not in locs) and checkToken(token):
                    filtered_tweet.add(token.lemma_.lower())
        
        if verbose:
            print(filtered_tweet, "\n---\n")
        
        # Filter out tweets that have too few words
        if len(filtered_tweet) >= MIN_TWEET_LEN:
            preprocessed_tweets.append(filtered_tweet)
            indices.append(index)
    
    return preprocessed_tweets, indices


# In[7]:


def convertClasses(c, indices):
    classes = []
    
    for index in indices:
        if pd.isnull(c[index]) or re.search(r'N', c[index]) is not None:
            classes.append(0)
        else:
            classes.append(1)
    
    return classes


# In[8]:


def preprocess(data, verbose=False):
    tweets = data['tweet']
    
    # Convert all to lowercase
    tweets = [splitWords(t) for t in tweets]
    
    # Process tweets through spaCy pipeline
    tweets, indices = spacyPipeline(tweets,verbose)
    
    # Transform with TF-IDF
    tfidf_tweets = TfidfVectorizer(max_df=0.8).fit_transform([' '.join(t) for t in tweets])
    
    # Transform classes
    if "existence" in data.columns:
        classes = convertClasses(data["existence"].tolist(), indices)
        return tfidf_tweets, classes

    return tfidf_tweets


# For Web Scraped tweets
# > Other notebooks call the preprocess function so I made different ones

# In[19]:


def preprocess2(data, verbose=False):
    tweets = data['tweet']
    
    # Convert all to lowercase
    tweets = [splitWords(t) for t in tweets]
    
    # Process tweets through spaCy pipeline
    tweets, indices = spacyPipeline(tweets,verbose)
    
    # Transform with TF-IDF (return fitted model)
    tf = TfidfVectorizer(max_df=0.8)
    fitted_transformer = tf.fit([' '.join(t) for t in tweets])
    tfidf_tweets = fitted_transformer.transform([' '.join(t) for t in tweets])

    # Transform classes
    if "existence" in data.columns:
        classes = convertClasses(data["existence"].tolist(), indices)
        return tfidf_tweets, classes, fitted_transformer

    return tfidf_tweets, classes, fitted_transformer


# In[15]:


def preprocess3(data, verbose=False, fitted_transformer=None):
    tweets = data['tweet']
    
    # Convert all to lowercase
    tweets = [splitWords(t) for t in tweets]
    
    # Process tweets through spaCy pipeline
    tweets, indices = spacyPipeline(tweets,verbose)
    
    # Transform with TF-IDF (use fitted model)
    tfidf_tweets = fitted_transformer.transform([' '.join(t) for t in tweets])
    
    # Transform classes
    if "existence" in data.columns:
        classes = convertClasses(data["existence"].tolist(), indices)
        return tfidf_tweets, classes, fitted_transformer

    return tfidf_tweets


# In[ ]:




