from textblob import TextBlob
import pandas as pd
import numpy as np
import nltk
import regex as re
import os 
import matplotlib.pyplot as plt

def preprocess(x):
    x = re.sub('[^a-z\s]', '', x.lower())
    x = [w for w in x.split()]
    return ' '.join(x)

def get_sentiment(review): 
        analysis = TextBlob(review) 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'

stop_words = set(nltk.corpus.stopwords.words("english"))

corpus = []
sentiment = []

list_of_files = os.listdir("movie reviews")

count = 0

for txt in list_of_files:
    with open("movie reviews/" + txt, 'r') as file:
        data = file.read().replace('\n', '')

        curr_words = []
        tokenized = nltk.tokenize.word_tokenize(data)
        for word in tokenized:
            pre = preprocess(word)
            if pre != '':
                if pre not in stop_words:
                    curr_words.append(pre)
        corpus.append(' '.join(curr_words))

        sentiment.append(get_sentiment(corpus[count]))
        count += 1






print(corpus[0:2])
print(sentiment[1000:1050])