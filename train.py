import sys
import os
import json
import pandas
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict

if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    csv_file = 'data/access.csv'

dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
dataset = dataframe.values

# Preprocess dataset
X = dataset[:,0]
Y = dataset[:,1]

tokenizer = Tokenizer(filters='\t\n', char_level=True)
tokenizer.fit_on_texts(X)

# Extract and save word dictionary
word_dict_file = 'build/word-dictionary.json'

if not os.path.exists(os.path.dirname(word_dict_file)):
    os.makedirs(os.path.dirname(word_dict_file))

with open(word_dict_file, 'w') as outfile:
    json.dump(tokenizer.word_index, outfile, ensure_ascii=False)

num_words = len(tokenizer.word_index)+1
X = tokenizer.texts_to_sequences(X)

max_log_length = 4096
train_size = int(len(dataset) * .75)

X_processed = sequence.pad_sequences(X, maxlen=max_log_length)
X_train, X_test = X_processed[0:train_size], X_processed[train_size:len(X_processed)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

model = Sequential()
model.add(Embedding(num_words, 32, input_length=max_log_length))
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=3, batch_size=256)

# Evaluate model
res = model.evaluate(X_test, Y_test, verbose=0)
print("Model Accuracy: {:0.2f}%".format(res[1] * 100))

# Save model
model.save_weights('securitai-lstm-weights.h5')
model.save('securitai-lstm-model.h5')
with open('securitai-lstm-model.json', 'w') as outfile:
    outfile.write(model.to_json())
