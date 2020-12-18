from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import nltk
import regex as re
import os 
import matplotlib.pyplot as plt
from textblob.sentiments import NaiveBayesAnalyzer

def preprocess(x):
    x = re.sub('[^a-z\s]', '', x.lower())
    x = [w for w in x.split()]
    return ' '.join(x)

def build_wordcloud(review, color):
    comment_words = '' 
    stopwords = set(STOPWORDS) 
    # for word in review: 
    #     # word = str(word) 
    #     tokens = word.split() 

    #     for i in range(len(tokens)): 
    #         tokens[i] = tokens[i].lower() 
        
    #     comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(review) 

    plt.figure(figsize = (8, 8), facecolor = color) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.show() 

sentiment_numbers = []

def get_sentiment(review): 
        analysis = TextBlob(review) 
        sentiment_numbers.append(analysis.sentiment.polarity)
        if analysis.sentiment.polarity > 0.05: 
            return 'positive'
        elif analysis.sentiment.polarity < -0.05: 
            return 'negative'
        else: 
            return 'neutral'



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

# plt.style.use('ggplot')

# x = ['positive', 'neutral', 'negative']
# y = [sentiment.count('positive'), sentiment.count('neutral'), sentiment.count('negative')]
# plt.bar(x,y)
# plt.show()



# plt.style.use('ggplot')

# x = ['positive', 'neutral', 'negative']
# y = [sentiment.count('positive'), sentiment.count('neutral'), sentiment.count('negative')]
# plt.bar(x,y)
# plt.show()



num_bins = 7
n, bins, patches = plt.hist(sentiment_numbers, num_bins, facecolor='blue', alpha=0.5)
plt.show()



positive_sentiment = []
negative_sentiment = []
neutral_sentiment = []

for i in range(len(sentiment)):
    if sentiment[i] == "positive":
        positive_sentiment.append(i)
    elif sentiment[i] == "negative":
        negative_sentiment.append(i)
    else:
        neutral_sentiment.append(i)

def build_review(indices, corpus):
    review = ""
    for i in indices:
        review += " " + corpus[i] + " "
    
    return review

positive_reviews = build_review(positive_sentiment, corpus)
negative_reviews = build_review(negative_sentiment, corpus)
neutral_reviews = build_review(neutral_sentiment, corpus)

build_wordcloud(positive_reviews, 'green')
build_wordcloud(negative_reviews, 'red')
build_wordcloud(neutral_reviews, 'blue')