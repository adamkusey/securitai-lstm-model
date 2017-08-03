import sys
import os
import pandas
import optparse

def analyze(csv_file):
    dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
    count_frame = dataframe.groupby([1]).count()
    print(count_frame)
    total_req = count_frame[0][0] + count_frame[0][1]
    num_malicious = count_frame[0][1]

    print("Malicious request logs in dataset: {:0.2f}%".format(float(num_malicious) / total_req * 100))

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/dev-access.csv'
    analyze(csv_file)
