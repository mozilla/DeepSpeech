#!/usr/bin/env python

"""
    NAME    : LDC TIMIT Dataset
    URL     : https://catalog.ldc.upenn.edu/ldc93s1
    HOURS   : 5
    TYPE    : Read - English
    AUTHORS : Garofolo, John, et al.
    TYPE    : LDC Membership
    LICENCE : LDC User Agreement
"""

import errno
import fnmatch
import os
import subprocess
import sys
import tarfile
from os import path

import pandas as pd


def clean(word):
    # LC ALL & strip punctuation which are not required
    new = word.lower().replace(".", "")
    new = new.replace(",", "")
    new = new.replace(";", "")
    new = new.replace('"', "")
    new = new.replace("!", "")
    new = new.replace("?", "")
    new = new.replace(":", "")
    new = new.replace("-", "")
    return new


def _preprocess_data(args):

    # Assume data is downloaded from LDC - https://catalog.ldc.upenn.edu/ldc93s1

    # SA sentences are repeated throughout by each speaker therefore can be removed for ASR as they will affect WER
    ignoreSASentences = True

    if ignoreSASentences:
        print("Using recommended ignore SA sentences")
        print(
            "Ignoring SA sentences (2 x sentences which are repeated by all speakers)"
        )
    else:
        print("Using unrecommended setting to include SA sentences")

    datapath = args
    target = path.join(datapath, "TIMIT")
    print(
        "Checking to see if data has already been extracted in given argument: %s",
        target,
    )

    if not path.isdir(target):
        print(
            "Could not find extracted data, trying to find: TIMIT-LDC93S1.tgz in: ",
            datapath,
        )
        filepath = path.join(datapath, "TIMIT-LDC93S1.tgz")
        if path.isfile(filepath):
            print("File found, extracting")
            tar = tarfile.open(filepath)
            tar.extractall(target)
            tar.close()
        else:
            print("File should be downloaded from LDC and placed at:", filepath)
            strerror = "File not found"
            raise IOError(errno, strerror, filepath)

    else:
        # is path therefore continue
        print("Found extracted data in: ", target)

    print("Preprocessing data")
    # We convert the .WAV (NIST sphere format) into MSOFT .wav
    # creates _rif.wav as the new .wav file
    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*.WAV"):
            sph_file = os.path.join(root, filename)
            wav_file = os.path.join(root, filename)[:-4] + "_rif.wav"
            print("converting {} to {}".format(sph_file, wav_file))
            subprocess.check_call(["sox", sph_file, wav_file])

    print("Preprocessing Complete")
    print("Building CSVs")

    # Lists to build CSV files
    train_list_wavs, train_list_trans, train_list_size = [], [], []
    test_list_wavs, test_list_trans, test_list_size = [], [], []

    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*_rif.wav"):
            full_wav = os.path.join(root, filename)
            wav_filesize = path.getsize(full_wav)

            # need to remove _rif.wav (8chars) then add .TXT
            trans_file = full_wav[:-8] + ".TXT"
            with open(trans_file, "r") as f:
                for line in f:
                    split = line.split()
                    start = split[0]
                    end = split[1]
                    t_list = split[2:]
                    trans = ""

                    for t in t_list:
                        trans = trans + " " + clean(t)

            # if ignoreSAsentences we only want those without SA in the name
            # OR
            # if not ignoreSAsentences we want all to be added
            if (ignoreSASentences and not ("SA" in os.path.basename(full_wav))) or (
                not ignoreSASentences
            ):
                if "train" in full_wav.lower():
                    train_list_wavs.append(full_wav)
                    train_list_trans.append(trans)
                    train_list_size.append(wav_filesize)
                elif "test" in full_wav.lower():
                    test_list_wavs.append(full_wav)
                    test_list_trans.append(trans)
                    test_list_size.append(wav_filesize)
                else:
                    raise IOError

    a = {
        "wav_filename": train_list_wavs,
        "wav_filesize": train_list_size,
        "transcript": train_list_trans,
    }

    c = {
        "wav_filename": test_list_wavs,
        "wav_filesize": test_list_size,
        "transcript": test_list_trans,
    }

    all = {
        "wav_filename": train_list_wavs + test_list_wavs,
        "wav_filesize": train_list_size + test_list_size,
        "transcript": train_list_trans + test_list_trans,
    }

    df_all = pd.DataFrame(
        all, columns=["wav_filename", "wav_filesize", "transcript"], dtype=int
    )
    df_train = pd.DataFrame(
        a, columns=["wav_filename", "wav_filesize", "transcript"], dtype=int
    )
    df_test = pd.DataFrame(
        c, columns=["wav_filename", "wav_filesize", "transcript"], dtype=int
    )

    df_all.to_csv(
        target + "/timit_all.csv", sep=",", header=True, index=False, encoding="ascii"
    )
    df_train.to_csv(
        target + "/timit_train.csv", sep=",", header=True, index=False, encoding="ascii"
    )
    df_test.to_csv(
        target + "/timit_test.csv", sep=",", header=True, index=False, encoding="ascii"
    )


if __name__ == "__main__":
    _preprocess_data(sys.argv[1])
    print("Completed")
