from nltk.tokenize import word_tokenize
from nltk import pos_tag
from autocorrect import spell
import re
import preprocessor as p

symbols = ['!', '@', '&', '#', '?', '...', '$', ':', ';', 'amp', '.', 'RT', ',']
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY)
def spell_check(inp):
    res = []
    for word in inp:
        res.append(spell(word))
    return res

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def tweet_prepro(inp):
    res = p.clean(inp)
    res = deEmojify(res)
    res = word_tokenize(res)
    res = [x for x in res if x not in symbols]
    return res

def get_pos(inp):
    return pos_tag(inp)
