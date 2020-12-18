from textblob import TextBlob
import pandas as pd
import numpy as np
import nltk
import regex as re

corpus = []


def preprocess(x):
    x = re.sub('[^a-z\s]', '', x.lower())
    x = [w for w in x.split()]
    return ' '.join(x)


