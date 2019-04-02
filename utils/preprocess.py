from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from autocorrect import spell


def prepro(inp):
    words = word_tokenize(inp)                 # tokenisasi kata
    res = []
    for w in words:
        w = spell (w)
        res.append(w)
    return res

def get_pos(inp):
    return pos_tag(inp)