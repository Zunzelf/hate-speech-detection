from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, Embedding, Bidirectional
from utils.dataset import Data
import pickle as pkl
from utils import preprocess as prepro
from utils.feature_extraction import WordEmbed as w2v

from keras.optimizers import RMSprop
#using keras
class BiLSTM():
	def create_model(self, num_words, max_words = 20, hidden_units = 150):
		model = Sequential()
		model.add(Embedding(num_words, 100, input_length = max_words, trainable = False))
		model.add(Bidirectional(LSTM(hidden_units)))
		model.add(Dropout(0.5))
		model.add(Dense(3, activation='softmax'))
		model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
		return model

# class LSTM():
# 	def create_model(self, num_words, max_words = 20, hidden_units = 150):
# 		model = Sequential()
# 		model.add(Embedding(num_words, 100, input_length = max_words, trainable = False))
# 		model.add(Bidirectional(LSTM(hidden_units)))
# 		model.add(Dropout(0.5))
# 		model.add(Dense(3, activation='softmax'))
# 		model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
# 		return model
class Driver():
    def load_model(self, path):
        with open(path, 'rb') as file:
            self.model = pkl.load(file)

    def load_word_model(self, path):
        with open(path, 'rb') as file:
            self.word_model = pkl.load(file)

    def predict(self, inp):
        tesx = inp
        tesx = prepro.clean(tesx)[:20]
        test = w2v().sen2vec_2(tesx, self.word_model)
        res = self.model.predict_classes(test)
        return res
    
    def train_model(self, x, y):
        pass
