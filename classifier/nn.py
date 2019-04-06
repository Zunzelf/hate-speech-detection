from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, Embedding, Bidirectional
from keras.optimizers import RMSprop
#using keras

class BiLSTM():
	def create_model(self, num_words, n_dims = 100, max_words = 20, hidden_units = 150, vectors = None):
		model = Sequential()
		model.add(Embedding(num_words, n_dims, input_length = max_words, trainable = False))
		model.add(Bidirectional(LSTM(hidden_units)))
		model.add(Dropout(0.5))
		model.add(Dense(3, activation='softmax'))
		model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
		return model

class DLSTM():
	def create_model(self, n_dims = 100, max_words = 20, hidden_units = 150):
		model = Sequential()
		model.add(LSTM(hidden_units, input_shape = ([n_dims * max_words, 1])))
		model.add(Dropout(0.5))
		model.add(Dense(3, activation='softmax'))
		model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
		return model

if __name__ == "__main__":
	import pickle as pkl
	from keras.preprocessing.text import Tokenizer
	from keras.preprocessing.sequence import pad_sequences

