from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os, csv, gensim, logging,pickle
import numpy as np

    
def word2idx(word, word_model):
    try :
        return word_model.wv.vocab[word].index
    except KeyError :
        return word_model.wv.vocab['$$'].index

def idx2word(idx, word_model):
    return word_model.wv.index2word[idx]

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

    def create_model(self, datas, size = 100, window = 5, min_count = 2):
        model = gensim.models.Word2Vec(
            datas,
            size = size,
            window = window,
            min_count = min_count,
            workers=10)
        model.train(datas, total_examples = len(datas), epochs=100)
        return model     
    
    def sen2vec(self, sentence, words = 10, length = 100, vectors = None):
        res = []
        x = 0
        zeros = np.zeros(length)
        if vectors != None:
            while x < words and x < len(sentence):
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

    def sen2vec_2(self, sentence, word_model):
        test = np.zeros([1, 20], dtype=np.int32)
        for t, word in enumerate(sentence):
            test[0, t] = word2idx(word, word_model)
        return test

    def save_model(self, model, path = 'sen2vec.mdl'):
        # save as pickle
        print("Save model to file...")
        pickle.dump(model, open(path, 'wb'))

    def load_model(self, path = 'sen2vec.mdl'):
        # load from pickle
        print("load model from file...")
        return pickle.load(open(path, 'rb'))

    def load_vectors(self, path):
        return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

if __name__ == "__main__":
    w2v = WordEmbed()
    import pickle as pkl
    with open('token-checked.bin', 'rb') as file:
        dats = pkl.load(file)
    print('>>>>', type(dats[0]))
    dats.extend([['$$'], ['$$']])
    model = w2v.create_model(dats)
    w2v.save_model(model)
    # model = w2v.load_vectors("GoogleNews-vectors.bin")
    print(model['i'])