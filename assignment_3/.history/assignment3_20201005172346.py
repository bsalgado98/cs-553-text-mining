import pandas as pd
import numpy as np
import nltk
import regex as re
from nltk.probability import FreqDist
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans

# nltk.download('popular')

stop_words = set(nltk.corpus.stopwords.words("english"))

# read in data
dataset = pd.read_csv('2019VAERSData.csv', encoding='latin-1')

# create serious variable
dataset["SERIOUS"] = np.where((dataset["DIED"] == "Y")
                              | (dataset["ER_VISIT"] == "Y") | (dataset["HOSPITAL"] == "Y")
                              | (dataset["DISABLE"] == "Y"), 'Y', 'N')


# preprocesses text
def preprocess(x):
    x = re.sub('[^a-z\s]', '', x)
    x = [w for w in x.split() if w not in stop_words]
    return ' '.join(x)


sentences = []
for row in dataset.itertuples():
    # make sure its a string
    if isinstance(row.SYMPTOM_TEXT, str):
        curr_words = []
        tokenized = nltk.tokenize.word_tokenize(row.SYMPTOM_TEXT)
        for word in tokenized:

            pre = preprocess(word)

            if pre != '':
                curr_words.append(pre)
        sentences.append(' '.join(curr_words))


vect = TfidfVectorizer(min_df=3)
tfidf_matrix = vect.fit_transform(sentences)
feature_names = vect.get_feature_names()

model = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
model.fit(vect)


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(2):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

# def get_ifidf_for_words(text):
#     tfidf_matrix = vect.transform(text).todense()
#     feature_index = tfidf_matrix[0, :].nonzero()[1]
#     tfidf_scores = zip([feature_names[i] for i in feature_index], [
#                        tfidf_matrix[0, x] for x in feature_index])
#     return dict(tfidf_scores)


# print("TF-IDF table")
# print({k: v for k, v in sorted(get_ifidf_for_words(
#     sentences).items(), key=lambda item: item[1], reverse=True)})
