import os, csv
try :
    from utils import preprocess
    from utils.feature_extraction import WordEmbed as w2v
except ModuleNotFoundError:
    import preprocess
    from feature_extraction import WordEmbed as w2v
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
# import nltk
# nltk.download('punkt')

class Data():
    def __init__(self):
        self.y = []      #   for data with class. Training
        self.x = []      #   for training feature
        
    def load_data(self, path):
        with open(path, mode ='r', encoding="utf-8") as csv_file:
            data = csv.reader(csv_file, delimiter=',')
            next(csv_file)
            for line in tqdm(data):
                self.x.append(line[2])
                self.y.append(int(line[1]))
            csv_file.close()
    
def tokenize_doc(docs, prepro = False):
    tokens = []
    for line in docs:
        if prepro:
            tmp = preprocess.prepro(line)
        elif not prepro:
            tmp = line.split(" ")
        tokens.append(tmp)
    return tokens

def tokenize_sen(doc): # for parallel process
    tmp = preprocess.tweet_prepro(doc)
    return tmp

if __name__ == "__main__":
    dat = Data()

    print("loading file...")
    dat.load_data('labeled_data.csv')
    print("loading file...complete!")

    print("Preparing Parallelism...", end = '')
    pool = mp.Pool(int(mp.cpu_count()))
    dats = dat.x
    print("complete! cpu ready : %i cpus" % int(mp.cpu_count()))

    print("tokenize docs...")
    tkn = list(tqdm(pool.imap(tokenize_sen, dats), total = len(dats)))
    pool.close
    print("tokenize docs...complete!")

    print("spell checking tokens...")
    pool = mp.Pool(int(mp.cpu_count()))    
    res = list(tqdm(pool.imap(preprocess.spell_check, tkn), total = len(tkn)))
    pool.close
    print("spell checking tokens...complete!")

    import pickle as pkl

    word2v = w2v()
    model =  word2v.load_model()
    vectorized = []
    for dat in tqdm(dats):
        vectorized.append(word2v.sen2vec(dats[0], vectors = model))

    vectorized = np.array(vectorized)
    vectorized = vectorized.reshape(vectorized.shape[0], -1, 1)  

    print(vectorized.shape)
    with open('token-vectorized.bin', 'wb') as file:
        pkl.dump(vectorized, file)

