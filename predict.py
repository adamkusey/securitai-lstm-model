import sys
import os
import json
import pandas
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict

if len(sys.argv) > 1:
    log_entry = sys.argv[1]
else:
    exit()

print log_entry
dataframe = pandas.read_csv('data/access.csv', engine='python', quotechar='|', header=None)
dataset = dataframe.values

# Preprocess dataset
X = dataset[:,0]

# Transform each log entry in X to include spaces
# This will allow us to easily parse the word dictionary
for index, item in enumerate(X):
    # Quick hack to space out json elements
    X[index] = json.dumps(json.loads(item, object_pairs_hook=OrderedDict), indent=1)

tokenizer = Tokenizer(num_words=None, filters='\t\n', split=' ', char_level=False)
tokenizer.fit_on_texts(X)

word_dict_file = 'build/word-dictionary.json'

if not os.path.exists(os.path.dirname(word_dict_file)):
    os.makedirs(os.path.dirname(word_dict_file))

with open(word_dict_file, 'w') as outfile:
    json.dump(tokenizer.word_index, outfile)

log_entry = json.dumps(json.loads(log_entry, object_pairs_hook=OrderedDict), indent=1)
print [log_entry]
seq = tokenizer.texts_to_sequences([log_entry])
print seq
max_log_length = 1024
log_entry_processed = sequence.pad_sequences(seq, maxlen=max_log_length)

model = load_model('securitai-lstm-model.h5')
model.load_weights('securitai-lstm-weights.h5')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
prediction = model.predict(log_entry_processed)
print prediction
