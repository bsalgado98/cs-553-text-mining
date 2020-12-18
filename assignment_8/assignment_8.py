from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
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

def build_wordcloud(review, sentiment):
    comment_words = '' 
    stopwords = set(STOPWORDS) 
    for word in review: 
        # word = str(word) 
        tokens = word.split() 

        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
        
        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.show() 

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

build_wordcloud(corpus[0], sentiment[0])

# print(corpus[0:2])
# print(sentiment[1000:1050])

