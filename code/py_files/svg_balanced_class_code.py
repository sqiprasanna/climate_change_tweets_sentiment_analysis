#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sqiprasanna/climate_change_tweets_sentiment_analysis/blob/main/code/SVG_code.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt 
from itertools import chain
from wordcloud import WordCloud, STOPWORDS 
# !pip install xlrd==1.2.0
import cufflinks as cf
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[4]:


get_ipython().run_line_magic('run', './svg_Preprocessing.ipynb')
get_ipython().run_line_magic('run', './svg_Visulaization.ipynb')


# In[109]:


# performing random over sampling to reduce the imbalance in classes 
# run the above cell before running this cell
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics.pairwise import cosine_similarity
tf_No_SVD = TfidfVectorizer()
transf_text_fit= tf_No_SVD.fit(data_new['cleaned_tweet'])
transf_text= transf_text_fit.transform(data_new['cleaned_tweet'].astype('U'))
print(transf_text)
print(transf_text.shape)
cos_sim=cosine_similarity(transf_text, transf_text)
print(cos_sim)


# In[90]:


X_train_nb_bal, X_test_nb_bal, y_train_nb_bal, y_test_nb_bal= train_test_split(transf_text, data_new['existence'], test_size=0.2, random_state=33)


# In[91]:


X_train_nb_bal.shape


# In[92]:


X_train_nb_bal,y_train_nb_bal = RandomOverSampler(random_state = 21).fit_resample(X_train_nb_bal,y_train_nb_bal)
X_train_nb_bal


# In[93]:


X_train_nb_bal.shape, y_train_nb_bal.shape


# In[110]:


import pickle
pickle.dump(transf_text_fit,open("tf_NO_SVD.pkl",'wb'))


# In[22]:


model_nb_bal = make_pipeline(MultinomialNB())
model_nb_bal.fit(X_train_nb_bal, y_train_nb_bal)
predicted_categories_NB_bal = model_nb_bal.predict(X_test_nb_bal)


# In[27]:


accu_nb_bal=round(accuracy_score(y_test_nb_bal,predicted_categories_NB_bal),3)
f1_score_nb_bal=round(f1_score(y_test_nb_bal,predicted_categories_NB_bal,average="binary",pos_label="Yes"),3)
Precision_nb_bal=round(precision_score(y_test_nb_bal,predicted_categories_NB_bal,average="binary",pos_label="Yes"),3)
Recall_nb_bal=round(recall_score(y_test_nb_bal,predicted_categories_NB_bal,average="binary",pos_label="Yes"),3)
print('Accuracy:',accu_nb_bal)
print('F1_Score:',f1_score_nb_bal)
print('Recall_Score:',Recall_nb_bal)
print('Precision_Score:',Precision_nb_bal)


# In[111]:


X_train_rf_bal, X_test_rf_bal, y_train_rf_bal, y_test_rf_bal= train_test_split(transf_text,data_new['existence'], test_size=0.2, random_state=33)


# In[112]:


X_train_rf_bal,y_train_rf_bal = RandomOverSampler(random_state = 21).fit_resample(X_train_rf_bal,y_train_rf_bal)


# In[113]:


model_random_forest = make_pipeline(RandomForestClassifier())
model_random_forest.fit(X_train_rf_bal, y_train_rf_bal)
predicted_categories_random_forest_bal = model_random_forest.predict(X_test_rf_bal)


# In[114]:


import pickle
pickle.dump(model_random_forest,open("model.pkl",'wb'))


# In[98]:


accu_rf=round(accuracy_score(y_test_rf_bal,predicted_categories_random_forest_bal),3)
f1_score_rf=round(f1_score(y_test_rf_bal,predicted_categories_random_forest_bal,average="binary",pos_label="Yes"), 3)
Precision_rf=round(precision_score(y_test_rf_bal,predicted_categories_random_forest_bal,average="binary",pos_label="Yes"),3)
Recall_rf=round(recall_score(y_test_rf_bal,predicted_categories_random_forest_bal,average="binary",pos_label="Yes"),3)
print('Accuracy:',accu_rf)
print('F1_Score:',f1_score_rf)
print('Recall_Score:',Recall_rf)
print('Precision_Score:',Precision_rf)


# In[30]:


Result_2={
                  'Model':['Random-Forest','Naive Bayes'],
                  'Accuracy(%)':[accu_rf*100,accu_nb_bal*100],
                  'F1_Score(%)':[f1_score_rf*100,f1_score_nb_bal*100],
                  'Precision(%)':[Precision_rf*100,Precision_nb_bal*100],
                  'Recall(%)':[Recall_rf*100,Recall_nb_bal*100]
        
                    }
Result_final=pd.DataFrame(Result_2)


# In[31]:


splot=Result_final.plot(x='Model',y=['Accuracy(%)','F1_Score(%)','Precision(%)','Recall(%)'],kind='bar',figsize=(15,10),cmap='Pastel2',width=0.9)
for p in splot.patches:
    splot.annotate(format(round(p.get_height()), '.0f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0,7),
                   textcoords='offset points')
plt.title('Evaluation Scores For Climate change Data for various Models')
plt.ylabel('Percentage')
plt.xlabel('Models')
plt.legend(loc='upper left')
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

