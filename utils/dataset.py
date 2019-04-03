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

if __name__ == "__main__":
    dat = Data()
    print("loading file...")
    dat.load_data('data.csv')
    print("loading file...complete!")

    print("Preparing Parallelism...", end = '')
    pool = mp.Pool(mp.cpu_count())
    dats = dat.x
    print("complete! cpu ready : %s cpus" % mp.cpu_count())

    print("tokenize docs...")
    tkn = [pool.apply(tokenize_doc, args = (x, True)) for x in tqdm(dats)]
    pool.close
    print("tokenize docs...complete!")
    
    print("saving tokens...", end = '')
    import pickle as pkl
    with open('token.bin', 'wb') as file:
        pkl.dump(tkn, file)
    print("complete!")
    # print(tkn)
