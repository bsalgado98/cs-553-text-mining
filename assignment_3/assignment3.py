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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
                pre = nltk.PorterStemmer().stem(pre)
                curr_words.append(pre)
        sentences.append(' '.join(curr_words))


vect = TfidfVectorizer(min_df=3)
tfidf_matrix = vect.fit_transform(sentences)
feature_names = vect.get_feature_names()

n = 20

model = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=1)
model.fit(tfidf_matrix)


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(n):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

dataset['cluster'] = model.labels_

clusters = dataset.groupby(['cluster', 'SERIOUS']).size()


fig, ax1 = plt.subplots(figsize = (26, 15))
sns.heatmap(clusters.unstack(level = 'SERIOUS'), ax = ax1, cmap = 'Reds')
ax1.set_xlabel('SERIOUS').set_size(2)
ax1.set_ylabel('cluster').set_size(n)
plt.show()

def make_array(num):
    new = []
    try:
        new.append(clusters[num]['Y'])
    except KeyError:
        new.append(0)
    try:
        new.append(clusters[num]['N'])
    except KeyError:
        new.append(0)
    return new


new_arr = []
for i in range(0, n):
    new_arr.append(make_array(i))
chi_array = np.array(new_arr)
print(chi_array)

chi2_stat, p_val, dof, ex = stats.chi2_contingency(chi_array)
print("===Chi2 Stat===")
print(chi2_stat)
print("===Degrees of Freedom===")
print(dof)
print("===P-Value===")
print(p_val)
print("===Contingency Table===")
print(ex)
