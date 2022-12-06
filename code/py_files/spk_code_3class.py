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


data['existence'] = data['existence'].fillna('neutral')


# In[5]:


data[data['existence']=="No"].tweet


# In[6]:


print(data['existence'].value_counts())


# In[7]:


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


# In[8]:


data['tweet']


# In[9]:


tweets = data["tweet"]
# tweets = tweets.drop_duplicates()
tweets


# In[10]:


# tweets[:100]


# In[11]:


# python -m spacy download en_core_web_sm
preprocessed_tweets,indices = preprocess_tweets(tweets)


# ## Modeling

# In[12]:


new_data = data.iloc[indices]


# In[13]:


prepr_tweets = [" ".join(each) for each in preprocessed_tweets]
new_data['cleaned_tweet'] = prepr_tweets
new_data.head()


# In[14]:


len(prepr_tweets), data.shape, new_data.shape


# In[15]:


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


# In[16]:


new_data.isna().sum()


# In[17]:


# compute similarity using cosine similarity
cos_sim=cosine_similarity(text_tf, text_tf)
print(cos_sim)


# In[18]:


# splitting data 

X_train, X_test, y_train, y_test = train_test_split(cos_sim, new_data['existence'], test_size=0.2, random_state=33)
print(" Test Data Shape:", X_test.shape)
print(" Train Data Shape:",X_train.shape)


# In[19]:


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


# In[20]:


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


# In[21]:


clfs = [
    svm.SVC(kernel='linear').fit(X_train, y_train),
#     LinearSVC().fit(X_train, y_train),
    DecisionTreeClassifier().fit(X_train, y_train),
    KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
]


# In[22]:


MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)


# In[ ]:


row_index = 0
for clf in clfs:
    print('===============================================\n')
    print("**********{}***********".format(clf.__class__.__name__))
    predicted = clf.predict(X_test)
#     tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    MLA_name = clf.__class__.__name__
    MLA_compare.loc[row_index,'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(clf.score(X_train, y_train), 2)
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round( accuracy_score(y_test,predicted), 2)
    MLA_compare.loc[row_index, 'MLA Precision'] = round( precision_score(y_test,predicted, average="macro"), 2)
    MLA_compare.loc[row_index, 'MLA Recall'] = round( recall_score(y_test,predicted, average="macro"), 2)
    MLA_compare.loc[row_index, 'MLA F1 Score'] = round( f1_score(y_test,predicted, average="macro"), 2)
    MLA_compare.loc[row_index, 'error_rate'] = round( 1-accuracy_score(y_test,predicted), 2)
    MLA_compare.loc[row_index, 'cross val score'] = cross_val_score(clf, cos_sim,new_data['existence'], cv=10).mean()
    row_index+=1


# In[ ]:





# In[ ]:


MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)
MLA_compare


# ## Random Under Sampling

# In[ ]:


from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler


# In[ ]:


# splitting data 

X_train, X_test, y_train, y_test = train_test_split(cos_sim, new_data['existence'], test_size=0.2, random_state=33)
print(" Test Data Shape:", X_test.shape)
print(" Train Data Shape:",X_train.shape)


# In[ ]:


X_train,y_train = RandomUnderSampler(random_state = 21).fit_resample(X_train,y_train)
# X_train,y_train = RandomOverSampler(random_state = 21).fit_resample(X_train,y_train)


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


under_MLA_columns = []
under_MLA_compare = pd.DataFrame(columns = under_MLA_columns)


# In[ ]:


under_clfs = [
    svm.SVC(kernel='linear').fit(X_train, y_train),
#     LinearSVC(C=0.0001).fit(X_train, y_train),
    DecisionTreeClassifier().fit(X_train, y_train),
    KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
]


# In[ ]:


row_index = 0
for clf in under_clfs:
    print('===============================================\n')
    print("**********{}***********".format(clf.__class__.__name__))
    predicted = clf.predict(X_test)
#     tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    under_MLA_name = clf.__class__.__name__
    under_MLA_compare.loc[row_index,'MLA Name'] = under_MLA_name
    under_MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(clf.score(X_train, y_train), 2)
    under_MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round( accuracy_score(y_test,predicted), 2)
    under_MLA_compare.loc[row_index, 'MLA Precision'] = round( precision_score(y_test,predicted, average="macro"), 2)
    under_MLA_compare.loc[row_index, 'MLA Recall'] = round( recall_score(y_test,predicted, average="macro"), 2)
    under_MLA_compare.loc[row_index, 'MLA F1 Score'] = round( f1_score(y_test,predicted, average="macro"), 2)
    under_MLA_compare.loc[row_index, 'error_rate'] = round( 1-accuracy_score(y_test,predicted), 2)
    under_MLA_compare.loc[row_index, 'cross val score'] = cross_val_score(clf, cos_sim,new_data['existence'], cv=10).mean()
    row_index+=1


# In[ ]:


under_MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)
under_MLA_compare


# ## Random Over Sampling

# In[ ]:


# splitting data 

X_train, X_test, y_train, y_test = train_test_split(cos_sim, new_data['existence'], test_size=0.2, random_state=33)
print(" Test Data Shape:", X_test.shape)
print(" Train Data Shape:",X_train.shape)


# In[ ]:


# Xtrain,ytrain = RandomUnderSampler(random_state = 21).fit_resample(X_train,y_train)
X_train,y_train = RandomOverSampler(random_state = 21).fit_resample(X_train,y_train)


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


over_MLA_columns = []
over_MLA_compare = pd.DataFrame(columns = over_MLA_columns)


# In[ ]:


over_clfs = [
    svm.SVC(kernel='linear').fit(X_train, y_train),
#     LinearSVC(C=0.0001).fit(X_train, y_train),

    DecisionTreeClassifier().fit(X_train, y_train),
    KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
]


# In[ ]:


row_index = 0
for clf in over_clfs:
    print('===============================================\n')
    print("**********{}***********".format(clf.__class__.__name__))
    predicted = clf.predict(X_test)
#     tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    over_MLA_name = clf.__class__.__name__
    over_MLA_compare.loc[row_index,'MLA Name'] = over_MLA_name
    over_MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(clf.score(X_train, y_train), 2)
    over_MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round( accuracy_score(y_test,predicted), 2)
    over_MLA_compare.loc[row_index, 'MLA Precision'] = round( precision_score(y_test,predicted, average="macro"), 2)
    over_MLA_compare.loc[row_index, 'MLA Recall'] = round( recall_score(y_test,predicted, average="macro"), 2)
    over_MLA_compare.loc[row_index, 'MLA F1 Score'] = round( f1_score(y_test,predicted, average="macro"), 2)
    over_MLA_compare.loc[row_index, 'error_rate'] = round( 1-accuracy_score(y_test,predicted), 2)
    over_MLA_compare.loc[row_index, 'cross val score'] = cross_val_score(clf, cos_sim,new_data['existence'], cv=10).mean()
    row_index+=1


# ## SMOTE

# In[ ]:


# splitting data 
X_train, X_test, y_train, y_test = train_test_split(cos_sim, new_data['existence'], test_size=0.2, random_state=33)
print(" Test Data Shape:", X_test.shape)
print(" Train Data Shape:",X_train.shape)


# In[ ]:


from imblearn.over_sampling import SMOTE

oversample = SMOTE(k_neighbors=5)
X_train,y_train = oversample.fit_resample(X_train,y_train)
X_train.shape,y_train.shape


# In[ ]:


SMOTE_MLA_columns = []
SMOTE_MLA_compare = pd.DataFrame(columns = SMOTE_MLA_columns)


# In[ ]:


smote_clfs = [
    svm.SVC(kernel='linear').fit(X_train, y_train),
#     LinearSVC(C=0.0001).fit(X_train, y_train),

    DecisionTreeClassifier().fit(X_train, y_train),
    KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
]


# In[ ]:


row_index = 0
for clf in smote_clfs:
    print('===============================================\n')
    print("**********{}***********".format(clf.__class__.__name__))
    predicted = clf.predict(X_test)
#     tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    SMOTE_MLA_name = clf.__class__.__name__
    SMOTE_MLA_compare.loc[row_index,'MLA Name'] = SMOTE_MLA_name
    SMOTE_MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(clf.score(X_train, y_train), 2)
    SMOTE_MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round( accuracy_score(y_test,predicted), 2)
    SMOTE_MLA_compare.loc[row_index, 'MLA Precision'] = round( precision_score(y_test,predicted, average="macro"), 2)
    SMOTE_MLA_compare.loc[row_index, 'MLA Recall'] = round( recall_score(y_test,predicted, average="macro"), 2)
    SMOTE_MLA_compare.loc[row_index, 'MLA F1 Score'] = round( f1_score(y_test,predicted, average="macro"), 2)
    SMOTE_MLA_compare.loc[row_index, 'error_rate'] = round( 1-accuracy_score(y_test,predicted), 2)
    SMOTE_MLA_compare.loc[row_index, 'cross val score'] = cross_val_score(clf, cos_sim,new_data['existence'], cv=10).mean()
    row_index+=1


# In[ ]:





# In[ ]:


MLA_compare


# In[ ]:


under_MLA_compare


# In[ ]:


over_MLA_compare


# In[ ]:


SMOTE_MLA_compare


# In[ ]:





# In[ ]:


# import pickle
# # pickling the vectorizer
# pickle.dump(tf, open('vectorizer', 'wb'))
# # pickling the model
# pickle.dump(clfs[0], open('svm_classifier_89', 'wb'))


# In[ ]:




