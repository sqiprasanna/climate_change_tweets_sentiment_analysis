#!/usr/bin/env python
# coding: utf-8

# In[32]:


import re
import numpy as np
import pandas as pd
import spacy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import enchant


# In[33]:


def checkWord(token):
    negations = ['not', 'no']
    fillers = ['link', 'http']
    return (not token.is_stop) and (token.is_alpha) and (token.text not in fillers) and ((token.text in negations) or (len(token.text) > 3))


# In[34]:


def checkToken(token):
    fillers = set({'link', 'http'})
    common = set() #set({'global', 'warming', 'climate', 'change'}) # Remove common signal between tweets
    return (len(token.text) > 3) and (not token.is_stop) and (token.is_alpha) and (token.text.lower() not in fillers) and (token.text.lower() not in common)


# In[35]:


def splitWords(t):
    splitTweet = []
    t = t.split(' ')
    
    for word in t:
        res = re.search(r'[A-Z]', word)
        if res is not None:
            ch = word[res.span()[0]]
            words = re.split(r'[A-Z]', word)
            splitTweet.extend([words[0], ch + words[1]])
        else:
            splitTweet.extend(re.split(r'["|,;!|:*]', word))
    
    return ' '.join(splitTweet)


# In[36]:


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


# In[37]:


def preprocess(tweets,verbose):

    whiteList=['climate','change','earth','global','warming','planet']

    #Remove mentions
    count=0
    mentions_tweets = [re.findall('@\w+', tweet) for tweet in tweets]

    #Remove hash sign
    hash_tweets = [re.findall('#\w+', tweet) for tweet in tweets]

    #Remove urls
    urls_tweets = [re.findall(r'http.?://[^\s]+[\s]?', tweet) for tweet in tweets]


    #Convert all to lowercase
    tweets = [t.lower() for t in tweets]
    
    # Process tweets through spaCy pipeline
    tweets, indices = spacyPipeline(tweets,False)
    
    # print(tweets)
    return tweets,hash_tweets,urls_tweets,mentions_tweets,indices

