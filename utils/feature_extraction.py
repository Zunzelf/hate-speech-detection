from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os, csv, gensim, logging,pickle
import numpy as np

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
    def get_feature(self, inp, vectors = None):
        # build vocabulary and train model
        return vectors[inp]

    def create_model(self, datas, size = 150, window = 10, min_count = 2):
        model = gensim.models.Word2Vec(
            datas,
            size = size,
            window = window,
            min_count = min_count,
            workers=10)
        model.train(datas, total_examples=len(datas), epochs=100)
        return model
        
    
    def sen2vec(self, sentence, words = 20, length = 150, vectors = None):
        res = []
        x = 0
        zeros = np.zeros(length)
        if vectors != None:
            while x < 20 and x < len(sentence):
                word = sentence[x]
                try:
                    vector = self.get_feature(word, vectors)
                except KeyError:
                    vector = zeros
                res.append(vector)
                x += 1
        gap = words - len(res)
        pad =  [zeros] * gap
        res.extend(pad)
        return np.array(res)

    def save_model(self, model, path = 'sen2vec.mdl'):
        # save as pickle
        print("Save model to file ............................")
        pickle.dump(model, open(path, 'wb'))

    def load_model(self, path = 'sen2vec.mdl'):
        # load from pickle
        print("load model from file .........................")
        return pickle.load(open(path, 'rb'))

    def load_vectors(self, path):
        return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

if __name__ == "__main__":
    # import pickle as pkl
    # with open('token-checked.bin', 'rb') as file:
    #     dats = pkl.load(file)
    w2v = WordEmbed()
    # model = w2v.create_model(dats)
    # w2v.save_model(model)
    model = w2v.load_vectors("GoogleNews-vectors.bin")
    print(model['i'])