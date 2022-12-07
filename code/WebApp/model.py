### Model.py

import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import spacy
import re
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
# from /Users/saivennelagarikapati/Downloads/257_ML_grp_13_project/code/svg_balanced_SVD_class_code.ipynb import tf

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
tf = pickle.load(open('tf.pkl','rb'))
# print(model.predict(["global warm report urg govern belgium world face increas hunger"]))

def checkWord(token):
    negations = ['not', 'no']
    fillers = ['link', 'http']
    return (not token.is_stop) and (token.is_alpha) and (token.text not in fillers) and ((token.text in negations) or (len(token.text) > 3))

def spacyPipeline(tweets):
    nlp = spacy.load('en_core_web_sm')
    
    MIN_TWEET_LEN = 5
    
    indices = []
    preprocessed_tweets = []
    for index, t in enumerate(tweets):
        
        doc = nlp(t)
        filtered_tweet = set()
        
        for token in doc:
            # print(token, " | ", spacy.explain(token.pos_))
            if (token.lemma_ not in filtered_tweet) and checkWord(token):
                filtered_tweet.add(token.lemma_.lower())
        
        if len(filtered_tweet) >= MIN_TWEET_LEN:
            print(filtered_tweet,"\n---\n")
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
    print(tweets)
    #Process tweets through spaCy pipeline
    tweets = spacyPipeline(tweets)
    print(tweets)
    
    #Filter out words
    tweets = [list(filter(lambda w: w != 'link', t)) for t in tweets]
    
    #Remove words less than length 2
    tweets = [list(filter(lambda w: len(w) > 2, t)) for t in tweets]

    # print(tweets)
    # print("ADITYA")
    return tweets

def sent_analysis(tweet: str) -> str:
    
    
    tweet = preprocess([tweet])
    if len(tweet) == 0:
        return ["Invalid Input"]
    
    
    tfidf_tweets = tf.transform([' '.join(t) for t in tweet])
    #cos_sim_rf=cosine_similarity(tfidf_tweets, tfidf_tweets)
    #print(cos_sim_rf)
    return model.predict(tfidf_tweets)