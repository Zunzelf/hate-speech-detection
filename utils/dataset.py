import os, csv
import preprocess
from tqdm import tqdm
import nltk
nltk.download('punkt')

class Data():
    def __init__(self):
        self.y = []    #   for data with class. Training
        self.x = []      #   for training feature
        
    def load_data(self, path):
        with open(path, mode ='r', encoding="utf-8") as csv_file:
            data = csv.reader(csv_file, delimiter=',')
            next(csv_file)
            print("loading file...")
            for line in tqdm(data):
                self.x.append(line[2])
                self.y.append(int(line[1]))

            csv_file.close()
    
    def tokenize_doc(self, prepro = False):
        tokens = []
        print("tokenize docs...")
        for line in tqdm(self.x):
            if prepro:
                tmp = preprocess.prepro(line)
            elif not prepro:
                tmp = line.split(" ")
            tokens.append(tmp)
        
        return tokens

if __name__ == "__main__":
    dat = Data()
    dat.load_data('data.csv')
    tkn = dat.tokenize_doc(True)
    import pickle as pkl
    with open('token.bin', 'rb') as file:
        pkl.dump(file, tkn)
    # print(tkn)