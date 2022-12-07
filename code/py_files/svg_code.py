#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sqiprasanna/climate_change_tweets_sentiment_analysis/blob/main/code/SVG_code.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[4]:


import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt 
from itertools import chain
from wordcloud import WordCloud, STOPWORDS 
# !pip install xlrd==1.2.0
import cufflinks as cf
import regex as re
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
# !pip install xlrd==1.2.0


# Collect the positive tweets that contain hashtags

# In[33]:


pre_tweet_pos,pos_hash_tweets,pos_urls_tweets,pos_mentions_tweets,indices_pos=preprocess(pos_tweets.tweet)


# In[34]:


generateWordCloud(pre_tweet_pos)


# Negative tweets visualization

# In[35]:


neg_tweets=data[data['existence']=='No']
neg_tweets.head()


# In[36]:


pre_tweet_neg,neg_hash_tweets,neg_urls_tweets,neg_mentions_tweets,indices_neg=preprocess (neg_tweets.tweet)


# In[37]:


generateWordCloud(pre_tweet_neg)


# In[38]:


# generateWordCloud([t.split(' ') for t in tweets])
generateWordCloud(preprocessed_tweets)


# In[39]:


import nltk
preprocessed_tweets
flat_list = []
for sublist in preprocessed_tweets:
    for item in sublist:
        flat_list.append(item)


# In[40]:


bigrams_series = (pd.Series(nltk.ngrams(flat_list, 2)).value_counts())[:10]


# In[41]:


bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
plt.title('10 Most Frequently Occuring Bigrams')
plt.ylabel('Bigram')
plt.xlabel('Number of Occurances')


# In[42]:


data['existence.confidence'].plot(
    kind='hist',
    bins=30,
    title='Tweet Sentiment Distribution ')


# In[43]:


data['word_count'].plot(
    kind='hist',
    # xTitle = "Word Count",
    bins=50,
    title='Tweet Word Count Distribution')


# In[44]:


print(data.groupby(by=["existence"]).sum())
data.groupby(by=["existence"]).sum().plot(kind = 'bar',title = "Existence Distribution")


# In[45]:


# The distribution of top unigrams before removing stop words
from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if len(word) > 3 and word not in set(STOPWORDS)]
    # words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(data['tweet'], 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['tweet' , 'word_count'])
display(df1.head())

df1.groupby('tweet').sum()['word_count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words in review before preprocessing')



# In[46]:


allwords = list(chain.from_iterable(preprocessed_tweets))
df2 = pd.DataFrame(allwords,columns = ["words"])
df2['words'].value_counts()[:20].plot(kind='bar', title='Top 20 words in review after preprocessing')
# allwords
# preprocessed_tweets


# In[47]:


data.head()


# In[48]:


prepr_tweets = [" ".join(each) for each in preprocessed_tweets]
data_new=data.iloc[indices_pre]
data_new['cleaned_tweet'] = prepr_tweets
data_new.head()


# In[49]:


len(prepr_tweets), data.shape


# ## TF IDF & modelling

# In[50]:


# TF IDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split,cross_validate
import collections, numpy
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score


# In[51]:


data.isna().sum()


# In[52]:


X_train, X_test, y_train, y_test_nb= train_test_split(data_new.cleaned_tweet,data_new['existence'], test_size=0.2, random_state=33)


# Use MultinomialNB

# In[53]:


model_nb = make_pipeline(TfidfVectorizer(), MultinomialNB())
model_nb.fit(X_train, y_train)
predicted_categories_NB = model_nb.predict(X_test)


# Create apickle file for webapp

# In[54]:


import pickle
pickle.dump(model_nb,open("model.pkl",'wb'))


# In[55]:


# plot the confusion matrix
mat = confusion_matrix(y_test_nb, predicted_categories_NB)
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=y_train.unique(),yticklabels=y_train.unique())
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()
print("The testing accuracy is {}".format(accuracy_score(y_test_nb, predicted_categories_NB)))
print("The training accuracy is {}".format(accuracy_score(y_train, model_nb.predict(X_train))))


# In[56]:


#Cross Validation Score


# In[57]:


from sklearn.model_selection import cross_val_score
scores_Naive_Bayes = cross_val_score(model_nb, data_new.cleaned_tweet,data_new['existence'], cv=10, scoring="accuracy")
print(scores_Naive_Bayes)
meanScore = scores_Naive_Bayes.mean()
print(meanScore * 100)


# Random Forest Imbalanced classes

# In[58]:


X_train, X_test, y_train, y_test = train_test_split(data_new.cleaned_tweet,data_new['existence'], test_size=0.3, random_state=33)


# In[59]:


from sklearn.ensemble import RandomForestClassifier
model_random_forest = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
model_random_forest.fit(X_train, y_train)
predicted_categories_random_forest = model_random_forest.predict(X_test)


# In[60]:


# plot the confusion matrix
mat = confusion_matrix(y_test, predicted_categories_random_forest)
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=y_train.unique(),yticklabels=y_train.unique())
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()
print("The testing accuracy is {}".format(accuracy_score(y_test, predicted_categories_random_forest)))
print("The training accuracy is {}".format(accuracy_score(y_train, model_random_forest.predict(X_train))))


# In[61]:


scores_random_forest = cross_val_score(model_random_forest, data_new.cleaned_tweet,data_new['existence'], cv=10, scoring="accuracy")
print(scores_random_forest)
meanScore = scores_random_forest.mean()
print(meanScore * 100)


# Comparisions between models

# In[62]:


results=[]
names=[]
results.append(scores_random_forest)
names.append("Random_Forest")


results.append(scores_Naive_Bayes)
names.append("Naive Bayes")



# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[63]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# In[64]:


accu_rf=round(accuracy_score(y_test,predicted_categories_random_forest),3)
f1_score_rf=round(f1_score(y_test,predicted_categories_random_forest,average="binary",pos_label="Yes"), 3)
Precision_rf=round(precision_score(y_test,predicted_categories_random_forest,average="binary",pos_label="Yes"),3)
Recall_rf=round(recall_score(y_test,predicted_categories_random_forest,average="binary",pos_label="Yes"),3)
print('Accuracy:',accu_rf)
print('F1_Score:',f1_score_rf)
print('Recall_Score:',Precision_rf)
print('Precision_Score:',Recall_rf)


# In[65]:


accu_nb=round(accuracy_score(y_test_nb,predicted_categories_NB),3)
f1_score_nb=round(f1_score(y_test_nb,predicted_categories_NB,average="binary",pos_label="Yes"),3)
Precision_nb=round(precision_score(y_test_nb,predicted_categories_NB,average="binary",pos_label="Yes"),3)
Recall_nb=round(recall_score(y_test_nb,predicted_categories_NB,average="binary",pos_label="Yes"),3)
print('Accuracy:',accu_nb)
print('F1_Score:',f1_score_nb)
print('Recall_Score:',Precision_nb)
print('Precision_Score:',Recall_nb)


# In[66]:


Result_2={
                  'Model':['Random-Forest','Naive Bayes'],
                  'Accuracy(%)':[accu_rf*100,accu_nb*100],
                  'F1_Score(%)':[f1_score_rf*100,f1_score_nb*100],
                  'Precision(%)':[Precision_rf*100,Precision_nb*100],
                  'Recall(%)':[Recall_rf*100,Recall_nb*100]
        
                    }
Result_final=pd.DataFrame(Result_2)


# In[67]:


splot=Result_final.plot(x='Model',y=['Accuracy(%)','F1_Score(%)','Precision(%)','Recall(%)'],kind='bar',figsize=(15,10),cmap='Pastel2',width=0.9)
for p in splot.patches:
    splot.annotate(format(round(p.get_height()), '.0f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0,7),
                   textcoords='offset points')
plt.title('Evaluation Scores For Climate change Data for various Models for imbalanced classes')
plt.ylabel('Percentage')
plt.xlabel('Models')
plt.legend(loc='upper left')
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)
plt.show()


# In[68]:


# performing random over sampling to reduce the imbalance in classes 
from imblearn.over_sampling import RandomOverSampler


# In[69]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
tf = TfidfVectorizer()
transf_text= tf.fit_transform(data_new['cleaned_tweet'].astype('U'))
transf_text_rf= TruncatedSVD(n_components=2500).fit_transform(transf_text)
transf_text_rf2= TruncatedSVD(n_components=2000).fit_transform(transf_text)
cos_sim=cosine_similarity(transf_text, transf_text)
cos_sim_rf=cosine_similarity(transf_text_rf, transf_text_rf)
cos_sim_rf2=cosine_similarity(transf_text_rf2, transf_text_rf2)
print(cos_sim)


# Applying naive Bayes for Balanced classes

# In[70]:


X_train_nb_bal, X_test_nb_bal, y_train_nb_bal, y_test_nb_bal= train_test_split(cos_sim,data_new['existence'], test_size=0.2, random_state=33)


# In[71]:


X_train_nb_bal.shape


# In[72]:


X_train_nb_bal,y_train_nb_bal = RandomOverSampler(random_state = 21).fit_resample(X_train_nb_bal,y_train_nb_bal)


# In[86]:


X_train_nb_bal.shape, y_train_nb_bal.shape


# In[85]:


model_nb_bal = make_pipeline(MultinomialNB())
model_nb_bal.fit(X_train_nb_bal, y_train_nb_bal)
predicted_categories_NB_bal = model_nb_bal.predict(X_test_nb_bal)


# In[84]:


accu_nb_bal=round(accuracy_score(y_test_nb_bal,predicted_categories_NB_bal),3)
f1_score_nb_bal=round(f1_score(y_test_nb_bal,predicted_categories_NB_bal,average="binary",pos_label="Yes"),3)
Precision_nb_bal=round(precision_score(y_test_nb_bal,predicted_categories_NB_bal,average="binary",pos_label="Yes"),3)
Recall_nb_bal=round(recall_score(y_test_nb_bal,predicted_categories_NB_bal,average="binary",pos_label="Yes"),3)
print('Accuracy:',accu_nb_bal)
print('F1_Score:',f1_score_nb_bal)
print('Recall_Score:',Precision_nb_bal)
print('Precision_Score:',Recall_nb_bal)


# Random Sampling Over sampler for Random Forest

# In[81]:


X_train_rf_bal, X_test_rf_bal, y_train_rf_bal, y_test_rf_bal= train_test_split(cos_sim_rf,data_new['existence'], test_size=0.2, random_state=33)


# In[82]:


X_train_rf_bal,y_train_rf_bal = RandomOverSampler(random_state = 21).fit_resample(X_train_rf_bal,y_train_rf_bal)


# In[83]:


model_random_forest = make_pipeline(RandomForestClassifier())
model_random_forest.fit(X_train_rf_bal, y_train_rf_bal)
predicted_categories_random_forest_bal = model_random_forest.predict(X_test_rf_bal)


# In[87]:


accu_rf=round(accuracy_score(y_test_rf_bal,predicted_categories_random_forest_bal),3)
f1_score_rf=round(f1_score(y_test_rf_bal,predicted_categories_random_forest_bal,average="binary",pos_label="Yes"), 3)
Precision_rf=round(precision_score(y_test_rf_bal,predicted_categories_random_forest_bal,average="binary",pos_label="Yes"),3)
Recall_rf=round(recall_score(y_test_rf_bal,predicted_categories_random_forest_bal,average="binary",pos_label="Yes"),3)
print('Accuracy:',accu_rf)
print('F1_Score:',f1_score_rf)
print('Recall_Score:',Precision_rf)
print('Precision_Score:',Recall_rf)


# In[88]:


Result_2={
                  'Model':['Random-Forest','Naive Bayes'],
                  'Accuracy(%)':[accu_rf*100,accu_nb_bal*100],
                  'F1_Score(%)':[f1_score_rf*100,f1_score_nb_bal*100],
                  'Precision(%)':[Precision_rf*100,Precision_nb_bal*100],
                  'Recall(%)':[Recall_rf*100,Recall_nb_bal*100]
        
                    }
Result_final=pd.DataFrame(Result_2)


# In[89]:


splot=Result_final.plot(x='Model',y=['Accuracy(%)','F1_Score(%)','Precision(%)','Recall(%)'],kind='bar',figsize=(15,10),cmap='Pastel2',width=0.9)
for p in splot.patches:
    splot.annotate(format(round(p.get_height()), '.0f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0,7),
                   textcoords='offset points')
plt.title('Evaluation Scores For Climate change Data for various Models for balanced classes')
plt.ylabel('Percentage')
plt.xlabel('Models')
plt.legend(loc='upper left')
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)


# SVD

# In[90]:


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


# In[77]:


from sklearn.metrics import make_scorer
from functools import partial
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


# In[91]:


X_train_rf_bal, X_test_rf_bal, y_train_rf_bal, y_test_rf_bal= train_test_split(cos_sim_rf,data_new['existence'], test_size=0.2, random_state=33)


# In[92]:


X_train_rf_bal,y_train_rf_bal = RandomOverSampler(random_state = 21).fit_resample(X_train_rf_bal,y_train_rf_bal)


# In[93]:


model_random_forest_SVD = make_pipeline(RandomForestClassifier())
model_random_forest_SVD.fit(X_train_rf_bal, y_train_rf_bal)
predicted_categories_random_forest_bal_SVD = model_random_forest.predict(X_test_rf_bal)


# In[95]:


import pickle
pickle.dump(model_random_forest_SVD,open("model.pkl",'wb'))


# In[96]:


cv_RF, mean_scores_RF = cross_val(X_train_rf_bal, y_train_rf_bal, X_test_rf_bal, y_test_rf_bal, RandomForestClassifier() )
print(cv_RF)
plot_cross_val(cv_RF)
mean_scores_RF


# SVD With 2000 components 

# In[ ]:


X_train_rf_bal, X_test_rf_bal, y_train_rf_bal, y_test_rf_bal= train_test_split(cos_sim_rf2,data_new['existence'], test_size=0.2, random_state=33)


# In[ ]:


X_train_rf_bal,y_train_rf_bal = RandomOverSampler(random_state = 21).fit_resample(X_train_rf_bal,y_train_rf_bal)


# In[ ]:


cv_RF, mean_scores_RF = cross_val(X_train_rf_bal, y_train_rf_bal, X_test_rf_bal, y_test_rf_bal, RandomForestClassifier() )
print(cv_RF)
plot_cross_val(cv_RF)
mean_scores_RF


# 
