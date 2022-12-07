# climate_change_tweets_sentiment_analysis 
### Project Title: Sentiment Analysis on Climate Change Using Twitter Tweets

### Team Members:
- John Lu (jfantab) (JohnLuSJSU was the account I used to upload initial notebooks/pdf)
- Sai Prasanna Kumar Kumaru (sqiprasanna)
- Sai Vennela Garikapati (Gsaivennela7)

### Introduction:


The project’s objective is to find out what percentage of Twitter users believe climate change exists. The main task is to classify the tweets into two categories whilst also comparing the performance of several different classification algorithms. This is also a good project to learn how to conduct preprocessing for an NLP task. This project is significant since it is a useful way to gauge public opinion on this issue. 

### Dataset:


The dataset is found from the link data.world/xprizeai-env/sentiment-of-climate-change and consists of 6,090 tweets in total about climate change or global warming [1]. The tweets are manually labeled by workers through the platform CrowdFlower [2]. The tweets are classified into three categories shown in the “existence” column: believing in climate change (yes), not believing in climate change (no), and not directly referencing climate change (null) [2]. The “existence_confidence” column shows the level of agreement for the judgment of that tweet and “the trust level of each of those workers” [2]. This third column will be ignored since we only care about the labels.

### Preprocessing:


An initial preprocessing step was removing duplicate tweets. The natural language preprocessing steps involved the spaCy pipeline, which comes pre-trained with various NLP preprocessing functionalities. spaCy provided the capabilities to remove stop words and to check for alphanumeric characters. We also converted all the words to lowercase and removed all words less than length two. Likewise, we removed the links, hashtags, and mentions, since they do not contribute to any meaning for climate change. However, we may experiment with keeping the hashtags in the future to see how it affects the sentiment analysis. 
In the future, we will transform the data using a TF-IDF vectorizer to properly weight the words based on their impact. However, we will need to investigate how well this works, since the tweets inherently have a small number of words.
Since this is text data, the number of features will be based on the number of unique words in all of the tweets. Therefore, we will need to perform dimensionality reduction techniques, such as TruncatedSVD from Sci-Kit Learn [2]. This method is favorable, since it does not center the data on its mean like PCA does [2].
One challenge in preprocessing is deciding which words to omit, since certain words that may look insignificant may have an actual impact on the sentiment analysis. Additionally, there is a possibility of misinterpretation for misspelled words in the tweets. Identifying the semantic meaning of words is challenging as there may be many synonyms for the word. 

### Method: 


The data will be segmented into train and test sets with an 80/20 split. We also plan on experimenting later on with cross-validation sets to prevent overfitting and make the best use of our dataset.
In order to perform sentiment analysis, we need to use various classification algorithms. We are planning on conducting linear classification or logistic regression on the one-hot encoded vectors of the tweets as initial models. These algorithms will respectively output either binary classes (+1, -1) or a probability (using a threshold to determine output class). In addition, we will also attempt other classification algorithms including Support Vector Machines or Naive Bayes. The best algorithm will be chosen by comparing the accuracy, precision, and recall on their respective performances.

### Steps to run the Web Application


Install all the required packages for the Web Application using
```
pip install -r requirements.txt
```

Start the server by running
```
python3 -m flask --app index run
```

Server should be started at 127.0.0.1:5000


