import pandas as pd
import numpy as np
import nltk
import regex as re
from nltk.probability import FreqDist
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

#nltk.download('popular')

stop_words = set(nltk.corpus.stopwords.words("english"))

# preprocesses text and makes everything lowercase
def preprocess(x):
    x = re.sub('[^a-z\s]', '', x.lower())                  
    x = [w for w in x.split() if w not in stop_words]
    return ' '.join(x) 
# preprocesses text
def preprocess2(x):
    x = re.sub('[^a-z\s]', '', x)                  
    x = [w for w in x.split() if w not in stop_words]
    return ' '.join(x) 
# used to do named entity extraction
def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(text))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        if current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk

# read in data 
dataset = pd.read_csv('2019VAERSData.csv', encoding='latin-1')

# create serious variable
dataset["SERIOUS"] = np.where((dataset["DIED"] == "Y" )
    | (dataset["ER_VISIT"] == "Y") | (dataset["HOSPITAL"] == "Y") 
    | (dataset["DISABLE"] == "Y"), 'Y', 'N')

# store frequencies 
all_words = FreqDist()
stemmed_words = FreqDist()
lem_words = FreqDist()
pos_count = FreqDist()
named_ent_count = FreqDist()


######  this section calculates needed info for deliverable 1 . 
###### WARNING - this will take 2-5 min to run

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
                lem_words[nltk.stem.WordNetLemmatizer().lemmatize(word, pos="v")] += 1
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
