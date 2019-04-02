from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from autocorrect import spell

def prepro(inp, stopword = False):
    if stopword:
        stopWords = set(stopwords.words('english'))
    words = word_tokenize(inp)                 # tokenisasi kata
    res = []
    for w in words:
        w = spell (w)
        if stopword:                           # word spell correction
            if w not in stopWords:                # stop words elimination
                res.append(w)
        elif not stopword:
            res.append(w)
    return res

def get_pos(inp):
    return pos_tag(inp)