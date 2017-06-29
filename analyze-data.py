import sys
import pandas
import numpy
numpy.random.seed(7)

if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    csv_file = 'data/access-sm.csv'

dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
#print dataframe
#print dataframe.iloc[:, 1].nunique()
ds = dataframe.groupby(1).nunique().values
normal_req = float(ds[0,0])
abnormal_req = float(ds[1,0])
print '% of malicious requests in dataset: ' + str(abnormal_req / normal_req * 100)

dataset = dataframe.values

# Preprocess dataset
X = dataset[:,0]
Y = dataset[:,1]
