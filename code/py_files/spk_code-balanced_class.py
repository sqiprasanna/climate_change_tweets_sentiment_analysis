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
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

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


data['existence'] = data['existence'].fillna('No')
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


# ## Modeling

# In[11]:


new_data = data.iloc[indices]


# In[12]:


prepr_tweets = [" ".join(each) for each in preprocessed_tweets]
new_data['cleaned_tweet'] = prepr_tweets
new_data.head()


# In[13]:


len(prepr_tweets), data.shape, new_data.shape


# In[14]:


# TF IDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import collections, numpy

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(new_data['cleaned_tweet'].astype('U'))

tf = TfidfVectorizer()
text_tf = tf.fit_transform(new_data['cleaned_tweet'].astype('U'))
# print(text_tf)


# In[15]:


new_data.isna().sum()


# In[16]:


# compute similarity using cosine similarity
cos_sim=cosine_similarity(text_tf, text_tf)
print(cos_sim)


# In[17]:


# splitting data 

X_train, X_test, y_train, y_test = train_test_split(cos_sim, new_data['existence'], test_size=0.2, random_state=33)
print(" Test Data Shape:", X_test.shape)
print(" Train Data Shape:",X_train.shape)


# In[18]:


pos = (y_test == 'Yes').sum()
neg = (y_test == 'No').sum()
postrain = (y_train == 'Yes').sum()
negtrain = (y_train == 'No').sum()
total = pos + neg
print(" Test Data Positive Sentiments :", pos)
print(" Test Data Negative Sentiments :",neg)
print(" Train Data Positive Sentiments :", postrain)
print(" Train Data Positive Sentiments :",negtrain)
new_data['existence'].value_counts()


# In[19]:


# perform algoritma KNN
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, auc,recall_score, f1_score,roc_curve,confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


# In[20]:


clfs = [
    svm.SVC(kernel='linear').fit(X_train, y_train),
#     LinearSVC().fit(X_train, y_train),
    DecisionTreeClassifier().fit(X_train, y_train),
    KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
]


# In[21]:


MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)


# In[22]:


row_index = 0
for clf in clfs:
    print('===============================================\n')
    print("**********{}***********".format(clf.__class__.__name__))
    predicted = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    MLA_name = clf.__class__.__name__
    MLA_compare.loc[row_index,'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(clf.score(X_train, y_train), 2)
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round( accuracy_score(y_test,predicted), 2)
    MLA_compare.loc[row_index, 'MLA Precision'] = round( precision_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    MLA_compare.loc[row_index, 'MLA Recall'] = round( recall_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    MLA_compare.loc[row_index, 'MLA F1 Score'] = round( f1_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    MLA_compare.loc[row_index, 'error_rate'] = round( 1-accuracy_score(y_test,predicted), 2)
    MLA_compare.loc[row_index, 'cross val score'] = cross_val_score(clf, cos_sim,new_data['existence'], cv=10).mean()
    plot_conf_matrix(y_test, predicted)
    row_index+=1


# In[23]:


print('MLA Precision', round( precision_score(y_test,predicted, average='macro'), 2))
print('MLA Recall', round( recall_score(y_test,predicted, average="macro"), 2))
print('MLA F1 Score', round( f1_score(y_test,predicted, average="macro"), 2))


# In[24]:


MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)
MLA_compare

