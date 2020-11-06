import pandas as pd
import numpy as np
import nltk
import regex as re
import math
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

corpus = [ "Democrat Joe Biden says, “No one is going to take our democracy away from us.” His comment came after President Donald Trump’s unfounded claims that Democrats were trying to “steal” the presidential election from him. In a Thursday evening tweet, Biden says, “America has come too far, fought too many battles, and endured too much to let that happen.”The nation is waiting to learn whether Biden or Trump will collect the 270 electoral votes needed to capture the presidency. Biden’s victories in Michigan and Wisconsin have put him in a commanding position, but Trump has showed no sign of giving up.",
"More votes are reported out of Pennsylvania. Trump’s lead in PA is now less than 50,000 votes with 94%\ of ballots counted, NBC News reports. Trump leads Biden 49.7% to 49.0%\ according to NBC. Many of the votes were reported out of Delaware County (outside Philadelphia) where Biden now leads Trump 62% to 37%.",
"In a separate lawsuit, the Trump campaign asked a judge to halt ballot counting in Pennsylvania, claiming that Republicans had been unlawfully denied access to observe the process. Meanwhile, Republicans in Pennsylvania have asked the U.S. Supreme Court to review a decision from the state’s highest court that allowed election officials to count mail-in ballots postmarked by Election Day that arrived through Friday.",
"Whether it's strange rashes on the toes or blood clots in the brain, the widespread ravages of COVID-19 have increasingly led researchers to focus on how the novel coronavirus sabotages the body's blood vessels.As scientists have come to know the disease better, they have homed in on the vascular system — the body's network of arteries, veins and capillaries, stretching more than 60,000 miles — to understand this wide-ranging disease and to find treatments that can stymie its most pernicious effects.",
"The ACTIV-5/BET study aims to streamline the pathway to finding urgently needed COVID-19 treatments by repurposing either licensed or late-stage-development medicines and testing them in a way that identifies the most promising agents for larger clinical studies in the most expedient way possible.",
"Hospitals and research labs all over the world are testing many different therapies on coronavirus-positive patients in an effort to find a potential COVID-19 treatment. Below we highlight a few medications and treatments that have been making a buzz in the science community.",
"These companies as they exist today have monopoly power. Some need to be broken up. All need to be properly regulated and held accountable, and adding that antitrust laws written a century ago need to be updated for the digital age.",
"The lawmakers say Congress should overhaul the laws that have let the companies grow so powerful. In particular, the report says, Congress should look at forcing structural separations of the companies and beefing up enforcement of existing antitrust laws.",
"US tech companies have faced increased scrutiny in Washington over their size and power in recent years. The investigation by the House Judiciary Committee is just one of multiple probes firms such as Facebook and Apple are facing."
]


stop_words = set(nltk.corpus.stopwords.words("english"))

def preprocess(x):
    x = re.sub('[^a-z\s]', '', x.lower())                  
    x = [w for w in x.split()]
    return ' '.join(x) 


sentences = []

stems = set()
stop_words_list = set()

for row in corpus:
    curr_words = []
    tokenized = nltk.tokenize.word_tokenize(row)
    for word in tokenized:
        pre = preprocess(word)
        if pre != '':

            if pre in stop_words:
                stop_words_list.add(pre)
            else:
                stemmed = nltk.PorterStemmer().stem(pre)
                if stemmed != pre:
                    stems.add(stemmed)
                curr_words.append(stemmed)
    sentences.append(' '.join(curr_words))

print('Stems')
print(stems)
print('Stop Words')
print(list(stop_words_list))


all_words = {}

def computeTFDict(words):
    """ Returns a tf dictionary for each words whose keys are all
    the unique words in the words and whose values are their
    corresponding tf.
    """
    # Counts the number of times the word appears in words
    wordsTFDict = {}
    for word in words.split(' '):
        
        if word in all_words:
            all_words[word] += 1
        else:
            all_words[word] = 1

        if word in wordsTFDict:
            wordsTFDict[word] += 1
        else:
            wordsTFDict[word] = 1
    # Computes tf for each word
    for word in wordsTFDict:
        wordsTFDict[word] = wordsTFDict[word] / len(words)
    return wordsTFDict

tfDict = []

for i in range(len(sentences)):

    tfDict.append(computeTFDict(sentences[i]))


popped_words = set()

for word in all_words:
    if all_words[word] == 1:
        popped_words.add(word)
        for temp in tfDict:
            temp.pop(word, "")

print('Popped Words')
print(popped_words)

all_words_list = list(all_words.keys())

print('TF')
print(tfDict)

def computeCountDict():
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of sentences in which
    the word appears.
    """
    countDict = {}
    # Run through each words's tf dictionary and increment countDict's (word, doc) pair
    for sentence in tfDict:
        for word in sentence:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict


countDict = computeCountDict()


def computeIDFDict():
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in countDict:
        idfDict[word] = math.log(len(sentences) / countDict[word])
    return idfDict
  

idfDict = computeIDFDict()


def computeTFIDFDict(wordsTFDict):
    """ Returns a dictionary whose keys are all the unique words in the
    words and whose values are their corresponding tfidf.
    """
    wordsTFIDFDict = {}
    #For each word in the words, we multiply its tf and its idf.
    for word in wordsTFDict:
        wordsTFIDFDict[word] = wordsTFDict[word] * idfDict[word]
    return wordsTFIDFDict


tfidfDict = [computeTFIDFDict(words) for words in tfDict]

print('TFIDF')
print(tfidfDict)


wordDict = sorted(countDict.keys())

def computeTFIDFVector(words):
    tfidfVector = [0.0] * len(wordDict)

    # For each unique word, if it is in the words, store its TF-IDF value.
    for i, word in enumerate(wordDict):
        if word in words:
            tfidfVector[i] = words[word]
    return tfidfVector

tfidfVector = [computeTFIDFVector(words) for words in tfidfDict]

npTfIDF = np.array(tfidfVector)

distOut = 1-pairwise_distances(npTfIDF, metric="cosine")

print('COSINE')
print(pd.DataFrame(distOut))

# def dot_product(vector_x, vector_y):
#     dot = 0.0
#     for e_x, e_y in zip(vector_x, vector_y):
#        dot += e_x * e_y
#     return dot


# def magnitude(vector):
#     mag = 0.0
#     for index in vector:
#       mag += math.pow(index, 2)
#     return math.sqrt(mag)


# review_similarity = dot_product(tfidfVector[0], tfidfVector[1]) /  magnitude(tfidfVector[0]) * magnitude(tfidfVector[1])

