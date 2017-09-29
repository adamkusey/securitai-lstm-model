## keras lstm rnn to perform binary classification on request logs.

As explained in [Detecting Malicious Requests Using Keras & Tensorflow](https://medium.com/slalom-engineering/detecting-malicious-requests-with-keras-tensorflow-5d5db06b4f28)

### python train.py [path to access.csv]

Training will split dataset into 75% train and 25% evaluation subsets. Model and metadata are saved upon completion.

### python predict.py [request log entry]

Loads saved model created from training to output confidence on given request entry.

#### Requirements:
- keras @ 2.0.5
- h5py
