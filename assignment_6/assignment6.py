import pandas as pd
import numpy as np
import nltk
import regex as re
from nltk.probability import FreqDist
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

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


def svd(): 
    vect = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf_matrix = vect.fit_transform(sentences).toarray()
    feature_names = vect.get_feature_names()

    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, dataset['SERIOUS'], test_size=0.2, random_state=0)

    print('SVD USED:')

    svd = TruncatedSVD(n_components=5)

    print('TF-IDF output shape:', X_train.shape)
        
    x_train_svd = svd.fit_transform(X_train)
    x_test_svd = svd.transform(X_test)
        
    print('SVD output shape:', x_train_svd.shape)
        
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Sum of explained variance ratio: %d%%" % (int(explained_variance * 100)))


    lr_model = LogisticRegression(solver='newton-cg',n_jobs=-1)
    lr_model.fit(x_train_svd, y_train)

    cv = KFold(n_splits=5, shuffle=True)
        
    scores = cross_val_score(lr_model, x_test_svd, y_test, cv=cv, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


    for i, comp in enumerate(svd.components_):
        terms_comp = zip(feature_names, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        print("Topic "+str(i)+": ")
        words = []
        for t in sorted_terms:
            words.append(t[0])
        print(words)


def build_lda(x_train, num_of_topic=5):
    print()
    print('LDA USED:')

    vec = CountVectorizer()
    count_matrix = vec.fit_transform(x_train)

    X_train, X_test, y_train, y_test = train_test_split(count_matrix, dataset['SERIOUS'], test_size=0.2, random_state=0)

    print('Count Vectorizer output shape:', X_train.shape)

    feature_names = vec.get_feature_names()

    lda = LatentDirichletAllocation(
        n_components=num_of_topic, max_iter=5, 
        learning_method='online', random_state=0)
    lda_x_train = lda.fit_transform(X_train)
    lda_x_test = lda.transform(X_test)


    print('LDA output shape:', lda_x_train.shape)

    lr_model = LogisticRegression(solver='newton-cg',n_jobs=-1)
    lr_model.fit(lda_x_train, y_train)

    cv = KFold(n_splits=5, shuffle=True)
    
    scores = cross_val_score(lr_model, lda_x_test, y_test, cv=cv, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    return lda, vec, feature_names

def display_word_distribution(model, feature_names, n_word):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        words = []
        for i in topic.argsort()[:-n_word - 1:-1]:
            words.append(feature_names[i])
        print(words)


svd()

lda_model, vec, feature_names = build_lda(sentences)
display_word_distribution(
    model=lda_model, feature_names=feature_names, 
    n_word=6)

