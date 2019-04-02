from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os, csv, gensim, logging,pickle
import numpy as np

class Data():
    def __init__(self):
        self.kelas = [] #for data with class. Training
        self.docs = [] #for training feature
        
    def load_data(self,filename):
        with open(filename, mode ='r', encoding="utf-8") as csv_file:
            data = csv.reader(csv_file, delimiter=',')
            next(csv_file)
            for line in data:
                self.docs.append(line[2])
                self.kelas.append(int(line[1]))

        csv_file.close()
        print(self.docs[2])
        # print(self.data[5])
    
    def tokenize_doc(self):
        token = []
        for line in self.docs:
            tmp = line.split(" ")
            token.append(tmp)
        
        return token

class TfIdf():
    def __init__(self, data):
        # belum ditokenisasi ya gapapa
        self.data = data
        self.vectorized = CountVectorizer()
        self.transformer = TfidfTransformer()

    def get_feature(self):
        X_counts = self.vectorized.fit_transform(self.data)

        # feature extraction
        X_tfidf = self.transformer.fit_transform(X_counts)

        return X_tfidf
    
class WordEmbed():
    def __init__(self, data = None, model = None):
        # data yang sudah ditokenisasi
        self.data = data
        self.model = model
    
    def get_feature(self, size = 150, window = 10, min_count = 2):
        # build vocabulary and train model
        model = gensim.models.Word2Vec(
            self.data,
            size=size,
            window=window,
            min_count=min_count,
            workers=10)
        model.train(self.data, total_examples=len(self.data), epochs=10)
        docs = np.array([self.sen2vec(x, vectors = model) for x in self.data])
        self.model = docs.reshape(docs.shape[0], -1, 1)
        # save to file
        self.save_model(self.model)

    
    def sen2vec(self, sentence, words = 20, length = 300, vectors = None):
        res = []
        x = 0
        zeros = np.zeros(length)
        if vectors != None:
            while x < 20 and x < len(sentence):
                word = sentence[x]
                try:
                    vector = vectors[word]
                except KeyError:
                    vector = zeros
                res.append(vector)
                x += 1
        gap = words - len(res)
        pad =  [zeros] * gap
        res.extend(pad)
        return np.array(res)

    def save_model(self, data, path = 'w2v_model/sen2vec.mdl'):
        # save as pickle
        print("Save model to file ............................")
        pickle.dump(data, open(path, 'wb'))

    def load_model(self, path = 'w2v_model/sen2vec.mdl'):
        # load from pickle
        print("load model from file .........................")
        self.model = pickle.load(open(path, 'rb'))

    def load_vectors(self, path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)