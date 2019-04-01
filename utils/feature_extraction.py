from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os,csv, gensim, logging

class Tf_idf():
    def __init__(self,data):
        # belum ditokenisasi ya gapapa
        self.data = data
        self.vectorized = CountVectorizer()
        self.transformer = TfidfTransformer()

    def get_feature(self):
        X_counts = self.vectorized.fit_transform(self.data)

        # feature extraction
        X_tfidf = self.transformer.fit_transform(X_counts)

        return X_tfidf
    
class Word_embed():
    def __init__(self,data):
        # data yang sudah ditokenisasi
        self.data = data
    
    def get_feature(self):
    # build vocabulary and train model
        model = gensim.models.Word2Vec(
            self.data,
            size=150,
            window=10,
            min_count=2,
            workers=10)
        model.train(self.data, total_examples=len(self.data), epochs=10)
        return model