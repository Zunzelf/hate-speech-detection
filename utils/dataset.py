import os, csv
import preprocess
import multiprocessing as mp
from tqdm import tqdm
# import nltk
# nltk.download('punkt')

class Data():
    def __init__(self):
        self.y = []    #   for data with class. Training
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
    tmp = preprocess.prepro(doc)
    return tmp

if __name__ == "__main__":
    dat = Data()
    print("loading file...")
    dat.load_data('data.csv')
    print("loading file...complete!")
    print("Preparing Parallelism...", end = '')
    pool = mp.Pool(int(mp.cpu_count()/2))
    # dats = dat.x[0:20000] #1
    # dats = dat.x[20001:30000] # 2
    # dats = dat.x[30001:40000]
    dats = dat.x[40001:50000] # 4
    # dats = dat.x[50001:-1]
    print("complete! cpu ready : %i cpus" % int(mp.cpu_count()/2))

    print("tokenize docs...")
    tkn = list(tqdm(pool.imap(tokenize_sen, dats), total = len(dats)))
    pool.close
    print("tokenize docs...complete!")
    
    print("saving tokens...", end = '')
    import pickle as pkl
    with open('token-4.bin', 'wb') as file:
        pkl.dump(tkn, file)
    print("complete!")
    # with open('token.bin', 'rb') as file:
    #     tkn = pkl.load(file)
    print(tkn)
