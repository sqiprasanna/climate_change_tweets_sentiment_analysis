### Model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import collections, numpy
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.stem import PorterStemmer
import spacy
import re
from spellchecker import SpellChecker

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
# print(model.predict(["global warm report urg govern belgium world face increas hunger"]))

def spacyPipeline(tweets):
    ps = PorterStemmer()
    nlp = spacy.load('en_core_web_sm')
    
    preprocessed_tweets = []
    for t in tweets:
        doc = nlp(t)
        filtered_tweet = []
        
        for token in doc:
            if (not token.is_stop) and token.is_alpha:
                filtered_tweet.append(ps.stem(str(token)))
        
        preprocessed_tweets.append(filtered_tweet)
    
    return preprocessed_tweets

def preprocess(tweets):

    spell = SpellChecker()
    fixed_tweets = []
    for tweet in tweets:
        valid = False
        for word in spell.split_words(tweet):
            if word == spell.correction(word):
                # Valid word
                valid = True
                break
        if valid:
            fixed_tweets.append(tweet)

    tweets = fixed_tweets
    
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
    
    #Process tweets through spaCy pipeline
    tweets = spacyPipeline(tweets)
    
    #Filter out words
    tweets = [list(filter(lambda w: w != 'link', t)) for t in tweets]
    
    #Remove words less than length 2
    tweets = [list(filter(lambda w: len(w) > 2, t)) for t in tweets]

    # print(tweets)
    return tweets

def sent_analysis(tweet: str) -> str:
    
    # d = enchant.Dict("en_US")
    
    # for tw in tweet:
    #     if not d.check(tw):
    #         wcount=wcount+1 
             
    #     tCount=tCount+1;
    
    tweet = preprocess([tweet])
    if len(tweet) == 0:
        return ["Invalid Input"]
    
    tweet = [" ".join(t) for t in tweet]
    return model.predict(tweet)