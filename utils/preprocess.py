from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from autocorrect import spell
from spellchecker import SpellChecker


def prepro(inp):
    words = word_tokenize(inp)                 # tokenisasi kata   
    spell = SpellChecker()

    res = []
    for x in words:
        word = x
        if spell.unknown([x]):
            word = spell.correction(x)
        res.append(word)
    return res

def get_pos(inp):
    return pos_tag(inp)