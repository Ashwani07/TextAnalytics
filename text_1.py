#%%
import pandas as pd
import time
import nltk

from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.corpus import words as wd


#%%
#Load the dataset
df_train = pd.read_csv("en_train.csv", encoding='UTF8')


#%%
df_train["sentence_id"].describe()
df_train.head()

df_train["before"].isnull().sum()
df_train[df_train["before"].isnull()]
df_train.loc[df_train["before"].isnull(), "before"] = "NA"
df_train.loc[df_train["after"].isnull(), "after"] = "NA"



#%%
#Create a sentence dataframe.
#train
before = df_train[["sentence_id", "before"]].groupby("sentence_id").before.apply(' '.join)
after = df_train[["sentence_id", "after"]].groupby("sentence_id").after.apply(' '.join)
train = pd.concat([before, after], axis =1)

# Merge all sentences
def cleanup(text):
    
    #lower case and split each word
    raw_text = text.lower().split()
    
    #Return the list without punctuations.
    return [w for w in raw_text if w not in punctuation]

#there are 748066 sentences
sentences = []
#For before Columns
start = time.time()
sentences = train.before.apply(lambda x: cleanup(x)).tolist()
print ("Time Taken", time.time() - start, "seconds")

#For after columns
start = time.time()
sentences +=  train.after.apply(lambda x: cleanup(x)).tolist()
print ("Time Taken", time.time() - start, "seconds")

#For test before Columns
start = time.time()
sentences +=  test.before.apply(lambda x: cleanup(x)).tolist()
print ("Time Taken", time.time() - start, "seconds")

#%%

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

#%%
# Abbreviation
from difflib import Differ
#import difflib
def abb_finder(raw_text, clean_text):
    l1 = raw_text.split()
    l2 = clean_text.split()
    dif = list(Differ().compare(l1, l2))
#    s = difflib.SequenceMatcher(None, raw_text, clean_text)
#    for tag, i1, i2, j1, j2 in s.get_opcodes():
#        print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(\
#              tag, i1, i2, j1, j2, raw_text[i1:i2], clean_text[j1:j2]))
    abb_dict ={}
    for i,x in enumerate(dif):
        
        if x[:2] == "- ":
            s = ""
            count = i+1
            while dif[count][:2] == "+ ":
                s += dif[count][2:] + " "
                count += 1
            if levenshteinDistance(x[2:], s) > 4:
                abb_dict[x[2:]] = s
    return abb_dict

abb_dict = train.apply(lambda row: abb_finder(row['before'],row['after']))

#%%

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)

from gensim.models import word2vec
print ("Training model...")

start = time.time()
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
print ("Time Taken", time.time() - start, "seconds")

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

#%%

words = model.wv.index2word

w_rank = {}
for i, word in enumerate(words):
    w_rank[word] = i

WORDS = w_rank

#%%
#PeterNOrvig Spell checker

def P(word): #, N=sum(WORDS.values())): 
    "Probability of `word`."
    #return WORDS[word] / N
    return WORDS.get(word, 0)

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


#%%
#Custom tagging each word
english_vocab = set(w.lower() for w in wd.words())
stop_words = stopwords.words('english')
#wn.synsets('motorcar')
def CustomTag(word):
    
    if word.isnumeric():
        return "Num"
    elif word.isalnum() and not word.isnumeric() and not word.isalpha() :
        return "Mixed"
    elif word in punctuation:
        return "Pun"
    elif word in english_vocab: #wn.synsets(word):
        return "Known"
    elif word in stop_words:
        return "SW"
    else:
        return "Unknown"

#%%
#For test data
df_test = pd.read_csv("en_test.csv", encoding='UTF8')

df_test.loc[df_test["before"].isnull(), "before"] = "NA"

before = df_test[["sentence_id", "before"]].groupby("sentence_id").before.apply(' '.join)
test = pd.concat([before], axis =1)

def modify(words):
    
    new = []
    for word in words:
        if word in abb_dict: new.append(abb_dict[word])
        
            