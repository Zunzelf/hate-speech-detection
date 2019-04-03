from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from autocorrect import spell


def prepro(inp):
    words = word_tokenize(inp)

    res = []
    for x in words:
        word = x
        spell(word)
        res.append(word)
    return res

def get_pos(inp):
    return pos_tag(inp)