from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Activation

from keras.optimizers import RMSprop
#using keras

def keras_model(X_train, Y_train, nb_classes, X_test = None, Y_test = None, hidden_units = 50):
	model = Sequential()
	model.add(LSTM(50, recurrent_activation="sigmoid", activation="tanh", input_shape=X_train.shape[1:], recurrent_initializer="glorot_uniform", unit_forget_bias=True))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	rmsprop = RMSprop(clipnorm = 5.0)
	model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
	return model

def train_model(model, X_train, Y_train, X_test = None, Y_test = None, epochs = 100, batch_size = 10, save_model = False):
	model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
	if save_model != False:
		import pickle as pkl
		with open(save_model, 'wb') as file:
			pkl.dump(model, file)