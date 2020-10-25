#%%

import keras
keras.__version__

#%%
from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#%%
print("train_data", train_data.shape)
print("train_labels", train_labels.shape)
print("test_data", test_data.shape)
print("test_labels", test_labels.shape)

print(train_data[10])
#%%

print("train_labels[10]", train_labels[10])

#%%
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

#%%
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
print("one_hot_train_labels", one_hot_train_labels.shape)
print("one_hot_test_labels", one_hot_test_labels.shape)

#%%
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Dense(units=64, activation="relu", input_shape = (10000, )))
model.add(layers.Dense(units=64, activation="relu"))
model.add(layers.Dense(units=46, activation="softmax"))
model.compile(optimizer =  "rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

#%%
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#%%
history = model.fit(x= partial_x_train, y=partial_y_train, batch_size=128, epochs=20, validation_data=(x_val, y_val))


#%%
print(partial_x_train.shape)
print(partial_y_train.shape)

