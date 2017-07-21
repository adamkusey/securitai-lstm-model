import sys
import os
import json
import pandas
import numpy
from keras.models import Sequential
from keras.models import load_model
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

dataframe = pandas.read_csv('data/dev-access.csv', engine='python', quotechar='|', header=None)
dataset = dataframe.values

# Preprocess dataset
X = dataset[:,0]

tokenizer = Tokenizer(filters='\t\n', char_level=True)
tokenizer.fit_on_texts(X)

log_entry = json.dumps(json.loads(log_entry, object_pairs_hook=OrderedDict), indent=1)
seq = tokenizer.texts_to_sequences([log_entry])
max_log_length = 4096
log_entry_processed = sequence.pad_sequences(seq, maxlen=max_log_length)

model = load_model('securitai-lstm-model.h5')
model.load_weights('securitai-lstm-weights.h5')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
prediction = model.predict(log_entry_processed)
print prediction[0]
