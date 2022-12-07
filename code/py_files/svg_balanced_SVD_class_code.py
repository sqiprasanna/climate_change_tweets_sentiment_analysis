#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sqiprasanna/climate_change_tweets_sentiment_analysis/blob/main/code/SVG_code.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS 
# !pip install xlrd==1.2.0
import cufflinks as cf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from sklearn.model_selection import cross_val_score
# !pip install xlrd==1.2.0
# TF IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split,cross_validate
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer


# In[2]:


get_ipython().run_line_magic('run', './svg_Preprocessing.ipynb')
get_ipython().run_line_magic('run', './svg_Visulaization.ipynb')


# ## TF IDF & modelling

# In[3]:


data.isna().sum()


# In[4]:


# performing random over sampling to reduce the imbalance in classes 
# RUN THE ABOVE CELL BEFORE RUNNING THIS

from imblearn.over_sampling import RandomOverSampler


# In[19]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
# tf = TfidfVectorizer()
tf = TfidfVectorizer().fit(data_new['cleaned_tweet'].astype('U'))
transf_text= tf.transform(data_new['cleaned_tweet'].astype('U'))
transf_text_rf= TruncatedSVD(n_components=2500).fit_transform(transf_text)
transf_text_rf2= TruncatedSVD(n_components=2000).fit_transform(transf_text)
cos_sim=cosine_similarity(transf_text, transf_text)
cos_sim_rf=cosine_similarity(transf_text_rf, transf_text_rf)
cos_sim_rf2=cosine_similarity(transf_text_rf2, transf_text_rf2)
print(cos_sim)


# In[20]:


import pickle
pickle.dump(tf,open("tf.pkl",'wb'))


# Random Sampling Over sampler for Random Forest SVD

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


def cross_val(X_train, Y_train, X_test, Y_test, model, folds=10):
    #scoring = ["accuracy", "precision","recall","f1_macro"]
    scoring = {'accuracy': make_scorer(accuracy_score),'precision': make_scorer(precision_score,pos_label="Yes"),'recall': make_scorer(recall_score,pos_label="Yes"),'f1_macro': make_scorer(f1_score,average="macro",pos_label="Yes") }
    cv = cross_validate(model, X_train, Y_train, cv=folds,error_score="raise", return_train_score=True, scoring=scoring, n_jobs=-1)
    
    train_scores = []
    test_scores = []
    
    for metric in scoring:
        train_scores.append(np.mean(cv[f"train_{metric}"]))
        test_scores.append(np.mean(cv[f"test_{metric}"]))
        
    mean_scores = pd.DataFrame((train_scores, test_scores), columns=scoring, index=["train", "test"])
    
    return cv, mean_scores


# In[21]:


X_train_rf_bal, X_test_rf_bal, y_train_rf_bal, y_test_rf_bal= train_test_split(cos_sim_rf,data_new['existence'], test_size=0.2, random_state=33)


# In[22]:


X_train_rf_bal,y_train_rf_bal = RandomOverSampler(random_state = 21).fit_resample(X_train_rf_bal,y_train_rf_bal)


# In[24]:


model_random_forest_SVD = make_pipeline(RandomForestClassifier())
model_random_forest_SVD.fit(X_train_rf_bal, y_train_rf_bal)
predicted_categories_random_forest_bal_SVD = model_random_forest_SVD.predict(X_test_rf_bal)


# In[25]:


import pickle
pickle.dump(model_random_forest_SVD,open("model.pkl",'wb'))


# In[12]:


cv_RF, mean_scores_RF = cross_val(X_train_rf_bal, y_train_rf_bal, X_test_rf_bal, y_test_rf_bal, RandomForestClassifier() )
print(cv_RF)
plot_cross_val(cv_RF)
mean_scores_RF


# SVD With 2000 components 

# In[13]:


X_train_rf_bal, X_test_rf_bal, y_train_rf_bal, y_test_rf_bal= train_test_split(cos_sim_rf2,data_new['existence'], test_size=0.2, random_state=33)


# In[14]:


X_train_rf_bal,y_train_rf_bal = RandomOverSampler(random_state = 21).fit_resample(X_train_rf_bal,y_train_rf_bal)


# In[15]:


cv_RF, mean_scores_RF = cross_val(X_train_rf_bal, y_train_rf_bal, X_test_rf_bal, y_test_rf_bal, RandomForestClassifier() )
print(cv_RF)
plot_cross_val(cv_RF)
mean_scores_RF


# 
