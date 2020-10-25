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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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


vect = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
tfidf_matrix = vect.fit_transform(sentences).toarray()
feature_names = vect.get_feature_names()

dataset['SERIOUS'][100:1000] = 'Y'

X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, dataset['SERIOUS'], test_size=0.2, random_state=0)




classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 



y_pred = classifier.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))