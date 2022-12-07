#!/usr/bin/env python
# coding: utf-8

# # Beautiful Soup Web Scraper

# Sources: 
# - https://www.freecodecamp.org/news/how-to-scrape-websites-with-python-2/
# - https://www.selenium.dev/documentation/webdriver/waits/
# - https://www.selenium.dev/documentation/webdriver/elements/finders/
# - https://stackoverflow.com/questions/73454187/how-to-get-tweets-in-twitter-using-selenium-in-python
# - https://pythonbasics.org/selenium-scroll-down/
# - https://javascript.info/size-and-scroll-window

# In[1]:


import time
import requests
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# In[ ]:


driver = webdriver.Chrome()
page = driver.get('https://twitter.com/search?q=%23globalwarming') # Getting page HTML through request
soup = BeautifulSoup(driver.page_source, 'html.parser') # Parsing content using beautifulsoup

sel = "div[data-testid='primaryColumn'] div[data-testid='tweetText']"

WebDriverWait(driver, 10).until(
    EC.visibility_of_all_elements_located(
        (By.CSS_SELECTOR, sel)
    )
)

allTweets = []
for i in range(50):
    driver.execute_script("window.scrollBy(0,document.body.scrollHeight * .15)")
    time.sleep(5)

    tweets = driver.find_elements(By.CSS_SELECTOR, sel)
    allTweets.extend([t.text for t in tweets])

driver.quit()


# In[115]:


df = pd.DataFrame(allTweets)
print(df.shape)
df.drop_duplicates(inplace=True, ignore_index=True)
print(df.shape)


# In[116]:


processed_tweets = []
for t in df[0]:
    s = ' '.join(t.split('\n'))
    processed_tweets.append(s)
print(len(processed_tweets))


# In[117]:


fh = open("tweets4.txt", "w")

for t in processed_tweets:
    fh.write(f"{t}\n")

fh.close()


# ## Twitter API

# Source: https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Tweet-Lookup/get_tweets_with_bearer_token.py

# In[31]:


import requests
import os
import json


# In[32]:


with open("secrets.json", "r") as f:
    secrets = json.load(f)

BEARER_TOKEN = secrets["bearerToken"]


# In[33]:


def create_url():
    query = "climate change"
    url = "https://api.twitter.com/2/tweets/search/recent?query={}".format(query)
    return url


# In[34]:


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r


# In[35]:


def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


# In[36]:


url = create_url()
json_response = connect_to_endpoint(url)

print(json.dumps(json_response, indent=4, sort_keys=True))
tweets = json_response


# In[ ]:




