keras lstm rnn to perform binary classification on request logs.

python train.py <path to access.csv>

Training will split dataset into 75% train and 25% evaluation subsets. Model and metadata are saved upon completion.

python predict.py <request log entry>

Loads saved model created from training to output confidence on given request entry.

Requirements:
- keras @ 2.0.5
- h5py
