import csv
import sys
import glob

'''
Usage: $ python3 check_characters.py "INFILE"
 e.g.  $ python3 ../DeepSpeech/util/check_characters.py "/home/data/*.csv" 
 e.g.  $ python3 ../DeepSpeech/util/check_characters.py "/home/data/french.csv" 
 e.g.  $ python3 ../DeepSpeech/util/check_characters.py "train.csv test.csv" 

Point this script to your transcripts, and it returns 
to the terminal the unique set of characters in those 
files (combined).

These files are assumed to be comma delimited, 
with the transcript being the third field.

The script simply reads all the text from all the files, 
storing a set of unique characters that were seen 
along the way.
'''

inFiles=sys.argv[1]
if "*" in inFiles:
    inFiles = glob.glob(inFiles)
else:
    inFiles = inFiles.split()

print("### Reading in the following transcript files: ###")
print(inFiles)

allText = set()
for inFile in (inFiles):
    with open(inFile, 'r') as csvFile:
        reader = csv.reader(csvFile)
        try:
            for row in reader:
                allText |= set(str(row[2]))
        except IndexError as ie:
            print("Your input file",inFile,"is not formatted properly. Check if there are 3 columns with the 3rd containing the transcript")
            sys.exit(-1)
        finally:
            csvFile.close()

print("### The following unique characters were found in your transcripts: ###")
print(list(allText))
print("### All these characters should be in your data/alphabet.txt file ###")
