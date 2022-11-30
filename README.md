# 257_ML_grp_13_project
### Project Title: Sentiment Analysis on Climate Change Using Twitter Tweets

### Team Members:
- John Lu (jfantab)
- Sai Prasanna Kumar Kumaru (sqiprasanna)
- Sai Vennela Garikapati (Gsaivennela7)

### Introduction:


Climate change has become significant with the dramatic increases in opinions regarding this issue, which play a key role in policy-making. This study proposes a method to analyze the dynamic opinions in the tweets and classify them into two categories—positive and negative. This helps in identifying the users’ opinions on environmental change. The project mainly focuses on finding out what percentage of Twitter users believe that there is a change in the climate. Once the data is trained using a supervised model, current views on climate change will be obtained through the Twitter API, and we can compare how the users' opinions have changed over the years.

### Dataset:


Dataset link: https://data.world/xprizeai-env/sentiment-of-climate-change. 
The tweets are collected using the Twitter API by searching for keywords or hashtags related to climate change or global warming [1]. The judgments (existence, categorical variable) and confidence scores (existence_confidence, percentages) are provided through a platform called CrowdFlower [1]. Humans were surveyed on whether they would classify the tweets as believing in climate change or not or unsure [1]. 

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

### Citations:


1. https://kbares.quora.com/Can-We-Figure-Out-If-Twitter-Users-Who-Discuss-Global-Warming-Believe-It-Is-Occurring 


2. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html 


