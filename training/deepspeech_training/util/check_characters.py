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
import csv
import os
import sys
import unicodedata

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-csv", "--csv-files", help="Str. Filenames as a comma separated list", required=True)
    parser.add_argument("-alpha", "--alphabet-format", help="Bool. Print in format for alphabet.txt", action="store_true")
    parser.add_argument("-unicode", "--disable-unicode-variants", help="Bool. DISABLE check for unicode consistency (use with --alphabet-format)", action="store_true")
    args = parser.parse_args()
    in_files = [os.path.abspath(i) for i in args.csv_files.split(",")]

    print("### Reading in the following transcript files: ###")
    print("### {} ###".format(in_files))

    all_text = set()
    for in_file in in_files:
        with open(in_file, "r") as csv_file:
            reader = csv.reader(csv_file)
            try:
                next(reader, None)  # skip the file header (i.e. "transcript")
                for row in reader:
                    if not args.disable_unicode_variants:
                        unicode_transcript = unicodedata.normalize("NFKC", row[2])
                        if row[2] != unicode_transcript:
                            print("Your input file", in_file, "contains at least one transript with unicode chars on more than one code-point: '{}'. Consider using NFKC normalization: unicodedata.normalize('NFKC', str).".format(row[2]))
                            sys.exit(-1)
                    all_text |= set(row[2])
            except IndexError:
                print("Your input file", in_file, "is not formatted properly. Check if there are 3 columns with the 3rd containing the transcript")
                sys.exit(-1)
            finally:
                csv_file.close()

    print("### The following unique characters were found in your transcripts: ###")
    if args.alphabet_format:
        for char in list(all_text):
            print(char)
        print("### ^^^ You can copy-paste these into data/alphabet.txt ###")
    else:
        print(list(all_text))

if __name__ == '__main__':
    main()
