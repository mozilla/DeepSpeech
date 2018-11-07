import csv
import sys
'''

Usage: $ python3 check_characters.py LOCALE
 e.g.  $ python3 ../DeepSpeech/util/check_characters.py 'fr' 

You need to run this script from the dir in which your csv files for test / train / dev are.
These files are comma delimited, with the transcript being the third field

'''

LOCALE=sys.argv[1]

allText=''

dev = 'cv_{}_valid_dev.csv'.format(LOCALE)
test = 'cv_{}_valid_test.csv'.format(LOCALE)
train = 'cv_{}_valid_train.csv'.format(LOCALE)

for inFile in (dev, test, train):
    with open(inFile, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            allText = ''.join(set(allText + str(row[2])))
    csvFile.close()

print("### The following characters were found in your train / test / dev transcripts: ###")
print(list(allText))
