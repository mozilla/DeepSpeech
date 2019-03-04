import csv
import sys
import glob

"""
Usage: $ python3 check_characters.py "INFILE"
 e.g.  $ python3 check_characters.py -csv /home/data/french.csv
 e.g.  $ python3 check_characters.py -csv ../train.csv,../test.csv 
 e.g.  $ python3 check_characters.py -alpha -csv ../train.csv 

Point this script to your transcripts, and it returns 
to the terminal the unique set of characters in those 
files (combined).

These files are assumed to be csv, with the transcript being the third field.

The script simply reads all the text from all the files, 
storing a set of unique characters that were seen 
along the way.
"""
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("-csv", "--csv-files", help="Str. Filenames as a comma separated list", required=True)
parser.add_argument("-alpha", "--alphabet-format",help="Bool. Print in format for alphabet.txt",action="store_true")
parser.set_defaults(alphabet_format=False)
args = parser.parse_args()
inFiles = [os.path.abspath(i) for i in args.csv_files.split(",")]

print("### Reading in the following transcript files: ###")
print("### {} ###".format(inFiles))

allText = set()
for inFile in (inFiles):
    with open(inFile, "r") as csvFile:
        reader = csv.reader(csvFile)
        try:
            next(reader, None)  # skip the file header (i.e. "transcript")
            for row in reader:
                allText |= set(str(row[2]))
        except IndexError as ie:
            print("Your input file",inFile,"is not formatted properly. Check if there are 3 columns with the 3rd containing the transcript")
            sys.exit(-1)
        finally:
            csvFile.close()

print("### The following unique characters were found in your transcripts: ###")
if args.alphabet_format:
    for char in list(allText):
        print(char)
    print("### ^^^ You can copy-paste these into data/alphabet.txt ###")
else:
    print(list(allText))
