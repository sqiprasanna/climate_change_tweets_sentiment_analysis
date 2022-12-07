#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay 

# %run ./jlu_preprocessing.ipynb


# ## Reading Data

# Original Data

# In[2]:


original_data = pd.read_csv('global_warming_tweets.csv', encoding= 'unicode_escape')
print(original_data.shape)
original_data.drop(columns=["existence.confidence"], inplace=True)
original_data.drop_duplicates(subset=["tweet"], inplace=True)

data = original_data

print(data.shape)
data.head()


# ## Data Exploration

# In[4]:


size = len(data["existence"])
cnt = [0, 0, 0]

for val in data["existence"]:
    if val == "N" or val == "No":
        cnt[0] += 1
    elif pd.isnull(val):
        cnt[1] += 1
    elif val == "Yes" or val == "Y":
        cnt[2] += 1

plt.figure(figsize=(3,3))
plt.xticks(np.arange(3), ["No", "N/A", "Yes"])
plt.xlabel("Classes")
plt.ylabel("Frequencies")
plt.title("Distribution of Classes")
plt.bar(np.arange(3), cnt)


# In[43]:


size = len(data["existence"])
cnt = [0, 0]

for val in data["existence"]:
    if pd.isnull(val) or val == "N" or val == "No":
        cnt[0] += 1
    elif val == "Yes" or val == "Y":
        cnt[1] += 1

plt.figure(figsize=(3,3))
plt.xticks(np.arange(2), ["No", "Yes"])
plt.xlabel("Classes")
plt.ylabel("Frequencies")
plt.title("Distribution of Classes")
plt.bar(np.arange(2), cnt)


# Running Algorithms

# > I borrowed Vennela's word cloud generation code and ran it again with my new preprocessing to get more accurate results.

# In[146]:


from itertools import chain
from wordcloud import WordCloud, STOPWORDS 

def generateWordCloud(tweets):
    allwords = " ".join(set(chain.from_iterable(tweets)))
    wordcloud = WordCloud(width = 600, height = 600, 
                    background_color ='white', 
                    stopwords = set(STOPWORDS), 
                    min_font_size = 10).generate(allwords)

    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.figure(figsize = (7, 7), facecolor = 'white', edgecolor='blue') 
    plt.imshow(wordcloud) 

    plt.show()


# In[147]:


def genCloud(data, fn):
    data = data[data["existence"].apply(fn)]
    tweets = data['tweet']

    # Convert all to lowercase
    tweets = [splitWords(t) for t in tweets]

    # Process tweets through spaCy pipeline
    tweets, indices = spacyPipeline(tweets)
    generateWordCloud(tweets)


# In[168]:


generateWordCloud([t.split(' ') for t in original_data['tweet']])


# In[165]:


genCloud(original_data, lambda x: x in ["Yes", "Y"])


# In[166]:


genCloud(data, lambda x: x in ["No", "N"])


# In[167]:


genCloud(data, lambda x: pd.isnull(x))


# ## Functions

# In[6]:


def cross_val(X_train, Y_train, X_test, Y_test, model, folds=10):
    scoring = ["accuracy", "precision", "recall", "f1"]
    cv = cross_validate(model, X_train, Y_train, cv=folds, return_train_score=True, scoring=scoring, n_jobs=-1)
    
    train_scores = []
    test_scores = []
    
    for metric in scoring:
        train_scores.append(np.mean(cv[f"train_{metric}"]))
        test_scores.append(np.mean(cv[f"test_{metric}"]))
        
    mean_scores = pd.DataFrame((train_scores, test_scores), columns=scoring, index=["train", "test"])
    
    return cv, mean_scores


# In[6]:


def plot_cross_val(cv, folds=10):    
    metrics = ["accuracy", "precision", "recall"]
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


# In[7]:


def pred(X_train, Y_train, X_test, Y_test, model, verbose=True):
    clf = model.fit(X_train, Y_train)
    pred = model.predict(X_train)
    pred2 = model.predict(X_test)
    
    if verbose:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

        conf_mat = confusion_matrix(y_true=Y_train, y_pred=pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        disp.plot(ax=ax1)    
        disp.ax_.set_title("Train Time")

        conf_mat2 = confusion_matrix(y_true=Y_test, y_pred=pred2)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_mat2)
        disp2.plot(ax=ax2)
        disp2.ax_.set_title("Test Time")
    
    return clf


# ## Testing

# Preprocessing

# > Note: I am combining the "No" and "N/A" classes together into one.

# In[27]:


get_ipython().run_line_magic('run', './jlu_preprocessing.ipynb')

X, Y = preprocess(data, verbose=False)
X


# In[28]:


svd = TruncatedSVD(n_components=2000)
svd = svd.fit(X)
X_ = svd.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_, Y, test_size=0.2)


# Testing

# In[8]:


model = pred(X_train, Y_train, X_test, Y_test, Perceptron())


# In[9]:


model = pred(X_train, Y_train, X_test, Y_test, LogisticRegression())


# Cross Validation

# In[10]:


# Perceptron

cv_perceptron, mean_scores_perceptron = cross_val(X_train, Y_train, X_test, Y_test, Perceptron())
plot_cross_val(cv_perceptron)
mean_scores_perceptron


# In[11]:


# Logistic Regression

cv_lr, mean_scores_lr = cross_val(X_train, Y_train, X_test, Y_test, LogisticRegression())
plot_cross_val(cv_lr)
mean_scores_lr


# Testing Web Scraped Tweets

# In[29]:


rawScrapedTweets = []

with open("scrapedTweets.txt") as fh:
    d = fh.read()
    rawScrapedTweets = d.split('\n')

scrapedTweets = pd.DataFrame(rawScrapedTweets, columns=["tweet"])

print(scrapedTweets.shape)
scrapedTweets.head()


# In[32]:


from imblearn.over_sampling import RandomOverSampler
get_ipython().run_line_magic('run', './jlu_preprocessing.ipynb')

tfidf_tweets_temp, _, fitted_transformer = preprocess2(data, verbose=False)
X_scraped = preprocess3(scrapedTweets, verbose=False, fitted_transformer=fitted_transformer)
X_scraped_ = svd.transform(X_scraped)
X_scraped_.shape


# In[33]:


from sklearn.ensemble import RandomForestClassifier
X_train_oversampled, Y_train_oversampled = RandomOverSampler(random_state = 21).fit_resample(X_train, Y_train)
rf_model = pred(X_train_oversampled, Y_train_oversampled, X_test, Y_test, RandomForestClassifier())


# In[34]:


rf_results = rf_model.predict(X_scraped_)


# In[35]:


cnt = [0, 0]
for cls in rf_results:
    if cls == 0:
        cnt[0] += 1
    else:
        cnt[1] += 1

plt.bar([0,1], cnt)
plt.xticks([0,1], ["No", "Yes"])


# Testing SVD

# In[134]:


def testSVD(X, Y, n_components):
    X_ = TruncatedSVD(n_components=n_components).fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_, Y, test_size=0.2)

    model = LogisticRegression()

    num_cv = 5
    
    cv = cross_validate(model, X_train, Y_train, cv=num_cv, return_train_score=True, n_jobs=-1)
    
    return np.sum(cv["test_score"])/num_cv


# In[53]:


X_svd_test, Y_svd_test = preprocess(data)
components = np.arange(1000,X_svd_test.shape[0]-(X_svd_test.shape[0]%1000),1000)
scores = []
for num in components:
    score = testSVD(X_svd_test, Y_svd_test, num)
    scores.append(score)
    
    print(num, " | ", score)
    
plt.figure(figsize=(4,4))
plt.plot(components, scores)
plt.ylabel("Test Accuracy")
plt.xlabel("Number of Components")
plt.title("Comparing Accuracies For Different Numbers of Components")
plt.show()


# Plot for Differing Numbers of Samples

# In[131]:


size = X.shape[0]
inc = 1000

samples = np.arange(0, size-(size%inc), inc)
acc_scores = []
prec_scores = []
rec_scores = []

original_data_ = data
for i in range(0, size-(size%inc), inc):
    data = original_data_.sample(n=i + inc, replace=False, axis=0)
    X_sampled, Y_sampled = preprocess(data)
    X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_sampled, Y_sampled, test_size=0.2)
    cv_lr, mean_scores_lr = cross_val(X_train_, Y_train_, X_test_, Y_test_, LogisticRegression())
    acc_scores.append(mean_scores_lr.loc["test","accuracy"])
    prec_scores.append(mean_scores_lr.loc["test","precision"])
    rec_scores.append(mean_scores_lr.loc["test","recall"])    

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))

sampling_scores = [acc_scores, prec_scores, rec_scores]
labels = ["Accuracy", "Precision", "Recall"]

for i in range(3):
    ax[i].plot(samples, sampling_scores[i])
    ax[i].set_title(labels[i])
    ax[i].set_xlabel("Number of Samples")

plt.tight_layout()
plt.show()


# In[ ]:




