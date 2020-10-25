#%%
import keras
keras.__version__

#%%
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#%%
print("train_data ", train_data.shape)
print("train_labels ", train_labels.shape)
print("test_data ", test_data.shape)
print("test_labels ", test_labels.shape)

#%%
train_labels[0]

#%%
max([max(sequence) for sequence in train_data])

#%%
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#%%

decoded_review
#%%
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

#%%
print("x_train", x_train.shape)
print("x_test", x_test.shape)

#%%
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#%%
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
#%%
from keras import layers
from keras import Sequential
from keras import models
model = models.Sequential()
model.add(layers.Dense(16, activation=keras.activations.relu, input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation=keras.activations.sigmoid))

#%%
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(x=partial_x_train, y=partial_y_train, batch_size=512, epochs=10, validation_data=(x_val, y_val))


#%%
model.evaluate(x_test, y=y_test)

#%%
history.history

#%% 
model.predict(x_test)