#In this model we'll be evaluating the IMDB datset using just one RNN, the Long Short Term Memory (LSTM) layer
#This model achieves a great validation accuracy of 89%, but is also significantly more 
#expensieve than many alternatives (approx 10 minutes on a GTX 980m)
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
from keras.layers import Dense
from keras.layers import LSTM
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.preprocessing import sequence

#Limiting the size of the data for the sake of efficiency
max_features = 10000
maxlen = 500
batch_size = 32

#Preparing our data 
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

model = Sequential()
model.add(Embedding(max_features, 32))
#The LSTM is a modified RNN layer that implements an additional information channel that backpropagates
#data from previous loops used in the model
#Additionally, this carried over data is transformed according to certain weights that work to 
#optimise the data e.g. by forgetting irrelevant information
#For a full mathematical explanation see: https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/lecture11.pdf
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
					epochs=10,
					batch_size=128,
					#New Parameter here, validation split means that the given fraction of the total 
		    			#data will be used as the validation data
					validation_split=0.2)

#End by plotting our data
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
