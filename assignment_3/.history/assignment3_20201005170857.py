import pandas as pd
import numpy as np
import nltk
import regex as re
from nltk.probability import FreqDist
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import collections
import networkx as nx
import matplotlib.pyplot as plt
nltk.download('popular')

stop_words = set(nltk.corpus.stopwords.words("english"))

# read in data
dataset = pd.read_csv('2019VAERSData.csv', encoding='latin-1')

# create serious variable
dataset["SERIOUS"] = np.where((dataset["DIED"] == "Y")
                              | (dataset["ER_VISIT"] == "Y") | (dataset["HOSPITAL"] == "Y")
                              | (dataset["DISABLE"] == "Y"), 'Y', 'N')

# store frequencies
all_words = FreqDist()
stemmed_words = FreqDist()
lem_words = FreqDist()
pos_count = FreqDist()
named_ent_count = FreqDist()


# this section calculates needed info for deliverable 1 .
# WARNING - this will take 2-5 min to run

words_storage = []

for row in dataset.itertuples():
    # make sure its a string
    if isinstance(row.SYMPTOM_TEXT, str):
        curr_words = []
        tokenized = nltk.tokenize.word_tokenize(row.SYMPTOM_TEXT)
        for word in tokenized:

            pre = preprocess2(word)

            if pre != '':
                curr_words.append(pre)

            word = preprocess(word)
            if word != "":
                all_words[word] += 1
                lem_words[nltk.stem.WordNetLemmatizer(
                ).lemmatize(word, pos="v")] += 1
                word = nltk.PorterStemmer().stem(word)
                stemmed_words[word] += 1
        tag = nltk.pos_tag(curr_words)
        for x, pos in tag:
            pos_count[pos] += 1
        for multi in get_continuous_chunks(tokenized):
            named_ent_count[multi] += 1
        words_storage.append(curr_words)

# creates word associations

finder = nltk.BigramCollocationFinder.from_documents(words_storage)

# only bigrams that appear 3+ times
finder.apply_freq_filter(3)

# return the 10 n-grams with the highest PMI
print("Top 10 word associations with PMI:")
print(finder.nbest(nltk.collocations.BigramAssocMeasures().pmi, 10))

# prints all results
print("All words:")
print(all_words.most_common)
print("Stemmed words:")
print(stemmed_words.most_common)
print("Lemmata words:")
print(lem_words.most_common)
print("Parts of speech:")
print(pos_count.most_common)
print("Named entities:")
print(named_ent_count.most_common)


# this section calculates part 1 for deliverable 2
# takes about 20 seconds

sentences = []
for row in dataset.itertuples():
    # make sure its a string
    if isinstance(row.SYMPTOM_TEXT, str):
        curr_words = []
        tokenized = nltk.tokenize.word_tokenize(row.SYMPTOM_TEXT)
        for word in tokenized:

            pre = preprocess2(word)

            if pre != '':
                curr_words.append(pre)
        sentences.append(' '.join(curr_words))


vect = TfidfVectorizer(min_df=3)
tfidf_matrix = vect.fit_transform(sentences)
feature_names = vect.get_feature_names()


def get_ifidf_for_words(text):
    tfidf_matrix = vect.transform(text).todense()
    feature_index = tfidf_matrix[0, :].nonzero()[1]
    tfidf_scores = zip([feature_names[i] for i in feature_index], [
                       tfidf_matrix[0, x] for x in feature_index])
    return dict(tfidf_scores)


print("TF-IDF table")
print({k: v for k, v in sorted(get_ifidf_for_words(
    sentences).items(), key=lambda item: item[1], reverse=True)})


# this section is part 2 of deliverable 2
# warning - takes 5-10 minutes

bgm = nltk.collocations.BigramAssocMeasures()
scored = finder.score_ngrams(bgm.likelihood_ratio)

# Group bigrams by first word in bigram.
prefix_keys = collections.defaultdict(list)
for key, scores in scored:
    prefix_keys[key[0]].append((key[1], scores))

# Sort keyed bigrams by strongest association.
for key in prefix_keys:
    prefix_keys[key].sort(key=lambda x: -x[1])

print("fever:")
print(prefix_keys['fever'][:15])

print("redness:")
print(prefix_keys['redness'][:15])
