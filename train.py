import sys
import os
import json
import pandas
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
numpy.random.seed(7)

if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    csv_file = 'data/access-sm.csv'

dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
dataset = dataframe.values

# Preprocess dataset
X = dataset[:,0]
Y = dataset[:,1]

# Transform each log entry in X to include spaces
# This will allow us to easily parse the word dictionary
for index, item in enumerate(X):
    # Quick hack to space out json elements
    X[index] = json.dumps(json.loads(item))

tokenizer = Tokenizer(num_words=None, filters='\t\n', split=' ', char_level=False)
tokenizer.fit_on_texts(X)

# Extract and save word dictionary
word_dict_file = 'build/word-dictionary.json'

if not os.path.exists(os.path.dirname(word_dict_file)):
    os.makedirs(os.path.dirname(word_dict_file))

with open(word_dict_file, 'w') as outfile:
    json.dump(tokenizer.word_index, outfile)

num_words = len(tokenizer.word_index)+1
X = tokenizer.texts_to_sequences(X)

max_log_length = 1024
train_size = int(len(dataset) * .75)
test_size = len(dataset) - train_size

X_processed = sequence.pad_sequences(X, maxlen=max_log_length)
X_train, X_test = X_processed[0:train_size], X_processed[train_size:len(X_processed)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# Create models
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(num_words, embedding_vector_length, input_length=max_log_length))
model.add(Dropout(0.2))
# LSTM specific dropout, try this if layer dropout isn't the best regularization.
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=3, batch_size=10)

# Evaluate model
res = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (res[1] * 100))
