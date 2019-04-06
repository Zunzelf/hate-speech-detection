from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, Embedding, Bidirectional


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
