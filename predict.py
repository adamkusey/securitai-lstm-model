import sys
import os
import json
import pandas
import numpy
import optparse
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict

def predict(csv_file, log_entry):
    # Loading processed word dictionary into keras Tokenizer would be better
    dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
    dataset = dataframe.values

    # Preprocess dataset
    X = dataset[:,0]
    for index, item in enumerate(X):
        reqJson = json.loads(item, object_pairs_hook=OrderedDict)
        del reqJson['timestamp']
        del reqJson['headers']
        del reqJson['source']
        del reqJson['route']
        del reqJson['responsePayload']
        X[index] = json.dumps(reqJson, separators=(',', ':'))

    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X)
    seq = tokenizer.texts_to_sequences([log_entry])
    max_log_length = 1024
    log_entry_processed = sequence.pad_sequences(seq, maxlen=max_log_length)

    model = load_model('securitai-lstm-model.h5')
    model.load_weights('securitai-lstm-weights.h5')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    prediction = model.predict(log_entry_processed)
    print prediction[0]

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/dev-access.csv'

    if args[0] is not None:
        predict(csv_file, args[0])
