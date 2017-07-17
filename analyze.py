import sys
import os
import json
import pandas
import numpy
from collections import OrderedDict

if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    csv_file = 'data/access.csv'

dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
count_frame = dataframe.groupby([1]).count()
print(count_frame)
total_req = count_frame[0][0] + count_frame[0][1]
num_malicious = count_frame[0][1]

print("Malicious request logs in dataset: {:0.2f}%".format(float(num_malicious) / total_req * 100))
