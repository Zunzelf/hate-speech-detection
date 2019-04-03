from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from autocorrect import spell
import re
from tqdm import tqdm

symbols = ['!', '@', '&', '#', '?', '...', '$', ':', ';', 'amp', '.']
def prepro(inp):
    words = " ".join([x for x in inp.split(" ") if 'http' not in x])
    words = word_tokenize(words)
    words = [x for x in words if x not in symbols]
    res = []
    for x in words:
        word = x
        res.append(spell(word))
    return res

def get_pos(inp):
    return pos_tag(inp)

if __name__ == "__main__":
    from time import time
    start = time()
    for b in tqdm(range(56000)):
        prepro("!!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out...")
    print(time()- start)