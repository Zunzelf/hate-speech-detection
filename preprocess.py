from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from autocorrect import spell

def normalize(sent):
    # stopWords = set(stopwords.words('english'))
    words = word_tokenize(sent)                 # tokenisasi kata
    wordsFiltered = []

    for w in words:
        w = spell (w)                           # word spell correction
        # if w not in stopWords:                # stop words elimination
        #     wordsFiltered.append(w)
        wordsFiltered.append(w)

    # tag = pos_tag(wordsFiltered)                # POS Tag

    print (wordsFiltered)