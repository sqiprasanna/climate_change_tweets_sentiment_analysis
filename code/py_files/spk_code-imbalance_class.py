#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


# from google.colab import drive
# drive.mount('/content/gdrive')


# In[4]:


# data = pd.read_excel(r'C:/Users/Checkout/Desktop/SJSU/sem1/257-ML/Project/global_warming_tweets.xls')
data = pd.read_csv(r'C:/Users/Checkout/Desktop/SJSU/sem1/257-ML/Project/global_warming_tweets_main.csv', engine='python') #encoding = "cp1252"
data.head()


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


# In[ ]:





# ## Functions

# In[12]:


def cross_val(X_train, Y_train, X_test, Y_test, model, folds=10):
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    cv = cross_validate(model, X_train, Y_train, cv=folds, return_train_score=True, scoring=scoring, n_jobs=-1, error_score = 'raise')
    
    train_scores = []
    test_scores = []
    
    for metric in scoring:
        train_scores.append(np.mean(cv[f"train_{metric}"]))
        test_scores.append(np.mean(cv[f"test_{metric}"]))
        
    mean_scores = pd.DataFrame((train_scores, test_scores), columns=scoring, index=["train", "test"])
    
    return cv, mean_scores


# In[13]:


def plot_cross_val(cv, folds=10):    
    metrics = ["accuracy", "precision_macro", "recall_macro",'f1_macro']
    colors = ["r", "g", "b", "y"]
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    
    for index, time_ in enumerate(["train", "test"]):
        for num, metric in enumerate(metrics):
            indexer = f"{time_}_{metric}"
            ax[index].plot(np.arange(folds), cv[indexer], color=colors[num])
        
        ax[index].legend(metrics, loc="upper left")
        ax[index].title.set_text(f"Scores during {time_} time")
    
    plt.tight_layout()
    plt.show()


# In[ ]:





# ## Modeling

# In[14]:


new_data = data.iloc[indices]


# In[15]:


prepr_tweets = [" ".join(each) for each in preprocessed_tweets]
new_data['cleaned_tweet'] = prepr_tweets
new_data.head()


# In[16]:


len(prepr_tweets), data.shape, new_data.shape


# In[17]:


# TF IDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import collections, numpy

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(new_data['cleaned_tweet'].astype('U'))

tf = TfidfVectorizer()
text_tf = tf.fit_transform(new_data['cleaned_tweet'].astype('U'))

text_tf_ = TruncatedSVD(n_components=2500).fit_transform(text_tf)

# print(text_tf)


# In[18]:


text_tf_.shape


# In[19]:


new_data.isna().sum()


# In[20]:


# compute similarity using cosine similarity
cos_sim=cosine_similarity(text_tf_, text_tf_)
print(cos_sim)


# In[21]:


cos_sim.shape


# In[22]:


# splitting data 

X_train, X_test, y_train, y_test = train_test_split(cos_sim, new_data['existence'], test_size=0.2, random_state=33)
print(" Test Data Shape:", X_test.shape)
print(" Train Data Shape:",X_train.shape)


# In[23]:


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


# In[24]:


# perform algoritma KNN
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, auc,recall_score, f1_score,roc_curve,confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score,cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


# In[25]:


models = [
    svm.SVC(kernel='linear'),
    DecisionTreeClassifier(),
    KNeighborsClassifier(n_neighbors=7)
]


# In[26]:


clfs = [
    svm.SVC(kernel='linear').fit(X_train, y_train),
#     LinearSVC().fit(X_train, y_train),
    DecisionTreeClassifier().fit(X_train, y_train),
    KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
]


# In[27]:


MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)


# In[28]:


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


# In[29]:


print('MLA Precision', round( precision_score(y_test,predicted, average='macro'), 2))
print('MLA Recall', round( recall_score(y_test,predicted, average="macro"), 2))
print('MLA F1 Score', round( f1_score(y_test,predicted, average="macro"), 2))


# In[30]:


MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)
MLA_compare


# In[ ]:


# cv_, mean_scores_ = cross_val(X_train, y_train, X_test, y_test, svm.SVC(kernel='linear'))


# In[ ]:


# mean_scores_


# In[39]:


# plot_cross_val(cv_)


# In[27]:


all_mean_scores_ = []
for model in models:
    print(model.__class__.__name__)
    cv_, mean_scores_ = cross_val(X_train, y_train, X_test, y_test, model)
    plot_cross_val(cv_)
    all_mean_scores_.append(mean_scores_)
    display(mean_scores_)


# In[28]:


# def testSVD(X, Y, n_components,model):
#     X_ = TruncatedSVD(n_components=n_components).fit_transform(X)
    
#     X_train, X_test, Y_train, Y_test = train_test_split(X_, Y, test_size=0.2)

#     num_cv = 5
    
#     cv = cross_validate(model, X_train, Y_train, cv=num_cv, return_train_score=True, n_jobs=-1)
    
#     return np.sum(cv["test_score"])/num_cv


# In[30]:


# X_svd_test, Y_svd_test = preprocess(data)


# In[35]:


# components = np.arange(1000,X_svd_test.shape[0]-(X_svd_test.shape[0]%1000),1000)
# components


# In[31]:


components = np.arange(1000,X_svd_test.shape[0]-(X_svd_test.shape[0]%1000),1000)
scores = []
for model in models:
    for num in components:
        print(model.__class__.__name__)
        score = testSVD(X_svd_test, Y_svd_test, num,model)
        scores.append(score)

        print(num, " | ", score)
    
# plt.figure(figsize=(4,4))
# plt.plot(components, scores)
# plt.ylabel("Test Accuracy")
# plt.xlabel("Number of Components")
# plt.title("Comparing Accuracies For Different Numbers of Components")
# plt.show()


# In[ ]:


#        accuracy  precision_macro  recall_macro  f1_macro
# train  0.923340         0.918074      0.879438  0.896318
# test   0.806289         0.755961      0.719149  0.732069


# ## SMOTE

# In[34]:


# splitting data 
X_train, X_test, y_train, y_test = train_test_split(cos_sim, new_data['existence'], test_size=0.2, random_state=33)
print(" Test Data Shape:", X_test.shape)
print(" Train Data Shape:",X_train.shape)


# In[35]:


from imblearn.over_sampling import SMOTE

oversample = SMOTE(k_neighbors=5)
X_train,y_train = oversample.fit_resample(X_train,y_train)
X_train.shape,y_train.shape


# In[36]:


SMOTE_MLA_columns = []
SMOTE_MLA_compare = pd.DataFrame(columns = SMOTE_MLA_columns)


# In[51]:


smote_clfs = [
    svm.SVC(kernel='linear').fit(X_train, y_train),
#     LinearSVC(C=0.0001).fit(X_train, y_train),

    DecisionTreeClassifier().fit(X_train, y_train),
    KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
]


# In[52]:


row_index = 0
for clf in smote_clfs:
    print('===============================================\n')
    print("**********{}***********".format(clf.__class__.__name__))
    predicted = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    SMOTE_MLA_name = clf.__class__.__name__
    SMOTE_MLA_compare.loc[row_index,'MLA Name'] = SMOTE_MLA_name
    SMOTE_MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(clf.score(X_train, y_train), 2)
    SMOTE_MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round( accuracy_score(y_test,predicted), 2)
    SMOTE_MLA_compare.loc[row_index, 'MLA Precision'] = round( precision_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    SMOTE_MLA_compare.loc[row_index, 'MLA Recall'] = round( recall_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    SMOTE_MLA_compare.loc[row_index, 'MLA F1 Score'] = round( f1_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    SMOTE_MLA_compare.loc[row_index, 'error_rate'] = round( 1-accuracy_score(y_test,predicted), 2)
    SMOTE_MLA_compare.loc[row_index, 'cross val score'] = cross_val_score(clf, cos_sim,new_data['existence'], cv=10).mean()
    print('MLA Precision', round( precision_score(y_test,predicted, average="macro"), 2))
    print('MLA Recall', round( recall_score(y_test,predicted, average="macro"), 2))
    plot_conf_matrix(y_test, predicted)

    print('MLA F1 Score', round( f1_score(y_test,predicted, average="macro"), 2))
    row_index+=1


# In[38]:


for model in models:
    cv_, mean_scores_ = cross_val(X_train, y_train, X_test, y_test, model)
    plot_cross_val(cv_)
    mean_scores_


# In[ ]:





# ## Random Under Sampling

# In[34]:


from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler


# In[35]:


# splitting data 

X_train, X_test, y_train, y_test = train_test_split(cos_sim, new_data['existence'], test_size=0.2, random_state=33)
print(" Test Data Shape:", X_test.shape)
print(" Train Data Shape:",X_train.shape)


# In[36]:


X_train,y_train = RandomUnderSampler(random_state = 21).fit_resample(X_train,y_train)
# X_train,y_train = RandomOverSampler(random_state = 21).fit_resample(X_train,y_train)


# In[37]:


X_train.shape, y_train.shape


# In[38]:


under_MLA_columns = []
under_MLA_compare = pd.DataFrame(columns = under_MLA_columns)


# In[39]:


under_clfs = [
    svm.SVC(kernel='linear').fit(X_train, y_train),
#     LinearSVC(C=0.0001).fit(X_train, y_train),
    DecisionTreeClassifier().fit(X_train, y_train),
    KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
]


# In[40]:


row_index = 0
for clf in under_clfs:
    print('===============================================\n')
    print("**********{}***********".format(clf.__class__.__name__))
    predicted = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    under_MLA_name = clf.__class__.__name__
    under_MLA_compare.loc[row_index,'MLA Name'] = under_MLA_name
    under_MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(clf.score(X_train, y_train), 2)
    under_MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round( accuracy_score(y_test,predicted), 2)
    under_MLA_compare.loc[row_index, 'MLA Precision'] = round( precision_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    under_MLA_compare.loc[row_index, 'MLA Recall'] = round( recall_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    under_MLA_compare.loc[row_index, 'MLA F1 Score'] = round( f1_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    under_MLA_compare.loc[row_index, 'error_rate'] = round( 1-accuracy_score(y_test,predicted), 2)
    print('MLA Precision', round( precision_score(y_test,predicted, average="macro"), 2))
    print('MLA Recall', round( recall_score(y_test,predicted, average="macro"), 2))
    print('MLA F1 Score', round( f1_score(y_test,predicted, average="macro"), 2))
    plot_conf_matrix(y_test, predicted)

    under_MLA_compare.loc[row_index, 'cross val score'] = cross_val_score(clf, cos_sim,new_data['existence'], cv=10).mean()
    row_index+=1


# In[63]:


under_MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)
under_MLA_compare


# ## Random Over Sampling

# In[41]:


# splitting data 

X_train, X_test, y_train, y_test = train_test_split(cos_sim, new_data['existence'], test_size=0.2, random_state=33)
print(" Test Data Shape:", X_test.shape)
print(" Train Data Shape:",X_train.shape)


# In[43]:


# Xtrain,ytrain = RandomUnderSampler(random_state = 21).fit_resample(X_train,y_train)
X_train,y_train = RandomOverSampler(random_state = 21).fit_resample(X_train,y_train)


# In[44]:


X_train.shape, y_train.shape


# In[45]:


over_MLA_columns = []
over_MLA_compare = pd.DataFrame(columns = over_MLA_columns)


# In[46]:


over_clfs = [
    svm.SVC(kernel='linear').fit(X_train, y_train),
#     LinearSVC(C=0.0001).fit(X_train, y_train),

    DecisionTreeClassifier().fit(X_train, y_train),
    KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
]


# In[47]:


row_index = 0
for clf in over_clfs:
    print('===============================================\n')
    print("**********{}***********".format(clf.__class__.__name__))
    predicted = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    over_MLA_name = clf.__class__.__name__
    over_MLA_compare.loc[row_index,'MLA Name'] = over_MLA_name
    over_MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(clf.score(X_train, y_train), 2)
    over_MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round( accuracy_score(y_test,predicted), 2)
    over_MLA_compare.loc[row_index, 'MLA Precision'] = round( precision_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    over_MLA_compare.loc[row_index, 'MLA Recall'] = round( recall_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    over_MLA_compare.loc[row_index, 'MLA F1 Score'] = round( f1_score(y_test,predicted, average="binary", pos_label="Yes"), 2)
    over_MLA_compare.loc[row_index, 'error_rate'] = round( 1-accuracy_score(y_test,predicted), 2)
    print('MLA Precision', round( precision_score(y_test,predicted, average="macro"), 2))
    print('MLA Recall', round( recall_score(y_test,predicted, average="macro"), 2))
    print('MLA F1 Score', round( f1_score(y_test,predicted, average="macro"), 2))
    over_MLA_compare.loc[row_index, 'cross val score'] = cross_val_score(clf, cos_sim,new_data['existence'], cv=10).mean()
    plot_conf_matrix(y_test, predicted)

    row_index+=1


# In[ ]:





# In[ ]:





# In[54]:


under_MLA_compare


# In[55]:


over_MLA_compare


# In[56]:


SMOTE_MLA_compare


# In[ ]:





# In[57]:


# import pickle
# # pickling the vectorizer
# pickle.dump(tf, open('vectorizer', 'wb'))
# # pickling the model
# pickle.dump(clfs[0], open('svm_classifier_89', 'wb'))

