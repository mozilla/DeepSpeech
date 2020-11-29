#!/usr/bin/env python
import codecs
import os
import re
import sys
import tarfile
import threading
import unicodedata
import urllib
from glob import glob
from multiprocessing.pool import ThreadPool
from os import makedirs, path

import pandas
from bs4 import BeautifulSoup
from tensorflow.python.platform import gfile
from deepspeech_training.util.downloader import maybe_download

"""The number of jobs to run in parallel"""
NUM_PARALLEL = 8

"""Lambda function returns the filename of a path"""
filename_of = lambda x: path.split(x)[1]


class AtomicCounter(object):
    """A class that atomically increments a counter"""

    def __init__(self, start_count=0):
        """Initialize the counter
        :param start_count: the number to start counting at
        """
        self.__lock = threading.Lock()
        self.__count = start_count

    def increment(self, amount=1):
        """Increments the counter by the given amount
        :param amount: the amount to increment by (default 1)
        :return:       the incremented value of the counter
        """
        self.__lock.acquire()
        self.__count += amount
        v = self.value()
        self.__lock.release()
        return v

    def value(self):
        """Returns the current value of the counter (not atomic)"""
        return self.__count


def _parallel_downloader(voxforge_url, archive_dir, total, counter):
    """Generate a function to download a file based on given parameters
    This works by currying the above given arguments into a closure
    in the form of the following function.

    :param voxforge_url: the base voxforge URL
    :param archive_dir:  the location to store the downloaded file
    :param total:        the total number of files to download
    :param counter:      an atomic counter to keep track of # of downloaded files
    :return:             a function that actually downloads a file given these params
    """

    def download(d):
        """Binds voxforge_url, archive_dir, total, and counter into this scope
        Downloads the given file
        :param d: a tuple consisting of (index, file) where index is the index
                  of the file to download and file is the name of the file to download
        """
        (i, file) = d
        download_url = voxforge_url + "/" + file
        c = counter.increment()
        print("Downloading file {} ({}/{})...".format(i + 1, c, total))
        maybe_download(filename_of(download_url), archive_dir, download_url)

    return download


def _parallel_extracter(data_dir, number_of_test, number_of_dev, total, counter):
    """Generate a function to extract a tar file based on given parameters
    This works by currying the above given arguments into a closure
    in the form of the following function.

    :param data_dir:       the target directory to extract into
    :param number_of_test: the number of files to keep as the test set
    :param number_of_dev:  the number of files to keep as the dev set
    :param total:          the total number of files to extract
    :param counter:        an atomic counter to keep track of # of extracted files
    :return:               a function that actually extracts a tar file given these params
    """

    def extract(d):
        """Binds data_dir, number_of_test, number_of_dev, total, and counter into this scope
        Extracts the given file
        :param d: a tuple consisting of (index, file) where index is the index
                  of the file to extract and file is the name of the file to extract
        """
        (i, archive) = d
        if i < number_of_test:
            dataset_dir = path.join(data_dir, "test")
        elif i < number_of_test + number_of_dev:
            dataset_dir = path.join(data_dir, "dev")
        else:
            dataset_dir = path.join(data_dir, "train")
        if not gfile.Exists(
            os.path.join(dataset_dir, ".".join(filename_of(archive).split(".")[:-1]))
        ):
            c = counter.increment()
            print("Extracting file {} ({}/{})...".format(i + 1, c, total))
            tar = tarfile.open(archive)
            tar.extractall(dataset_dir)
            tar.close()

    return extract


def _download_and_preprocess_data(data_dir):
    # Conditionally download data to data_dir
    if not path.isdir(data_dir):
        makedirs(data_dir)

    archive_dir = data_dir + "/archive"
    if not path.isdir(archive_dir):
        makedirs(archive_dir)

    print(
        "Downloading Voxforge data set into {} if not already present...".format(
            archive_dir
        )
    )

    voxforge_url = "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit"
    html_page = urllib.request.urlopen(voxforge_url)
    soup = BeautifulSoup(html_page, "html.parser")

    # list all links
    refs = [l["href"] for l in soup.find_all("a") if ".tgz" in l["href"]]

    # download files in parallel
    print("{} files to download".format(len(refs)))
    downloader = _parallel_downloader(
        voxforge_url, archive_dir, len(refs), AtomicCounter()
    )
    p = ThreadPool(NUM_PARALLEL)
    p.map(downloader, enumerate(refs))

    # Conditionally extract data to dataset_dir
    if not path.isdir(os.path.join(data_dir, "test")):
        makedirs(os.path.join(data_dir, "test"))
    if not path.isdir(os.path.join(data_dir, "dev")):
        makedirs(os.path.join(data_dir, "dev"))
    if not path.isdir(os.path.join(data_dir, "train")):
        makedirs(os.path.join(data_dir, "train"))

    tarfiles = glob(os.path.join(archive_dir, "*.tgz"))
    number_of_files = len(tarfiles)
    number_of_test = number_of_files // 100
    number_of_dev = number_of_files // 100

    # extract tars in parallel
    print(
        "Extracting Voxforge data set into {} if not already present...".format(
            data_dir
        )
    )
    extracter = _parallel_extracter(
        data_dir, number_of_test, number_of_dev, len(tarfiles), AtomicCounter()
    )
    p.map(extracter, enumerate(tarfiles))

    # Generate data set
    print("Generating Voxforge data set into {}".format(data_dir))
    test_files = _generate_dataset(data_dir, "test")
    dev_files = _generate_dataset(data_dir, "dev")
    train_files = _generate_dataset(data_dir, "train")

    # Write sets to disk as CSV files
    train_files.to_csv(os.path.join(data_dir, "voxforge-train.csv"), index=False)
    dev_files.to_csv(os.path.join(data_dir, "voxforge-dev.csv"), index=False)
    test_files.to_csv(os.path.join(data_dir, "voxforge-test.csv"), index=False)


def _generate_dataset(data_dir, data_set):
    extracted_dir = path.join(data_dir, data_set)
    files = []
    for promts_file in glob(os.path.join(extracted_dir + "/*/etc/", "PROMPTS")):
        if path.isdir(os.path.join(promts_file[:-11], "wav")):
            with codecs.open(promts_file, "r", "utf-8") as f:
                for line in f:
                    id = line.split(" ")[0].split("/")[-1]
                    sentence = " ".join(line.split(" ")[1:])
                    sentence = re.sub("[^a-z']", " ", sentence.strip().lower())
                    transcript = ""
                    for token in sentence.split(" "):
                        word = token.strip()
                        if word != "" and word != " ":
                            transcript += word + " "
                    transcript = (
                        unicodedata.normalize("NFKD", transcript.strip())
                        .encode("ascii", "ignore")
                        .decode("ascii", "ignore")
                    )
                    wav_file = path.join(promts_file[:-11], "wav/" + id + ".wav")
                    if gfile.Exists(wav_file):
                        wav_filesize = path.getsize(wav_file)
                        # remove audios that are shorter than 0.5s and longer than 20s.
                        # remove audios that are too short for transcript.
                        if (
                            (wav_filesize / 32000) > 0.5
                            and (wav_filesize / 32000) < 20
                            and transcript != ""
                            and wav_filesize / len(transcript) > 1400
                        ):
                            files.append(
                                (os.path.abspath(wav_file), wav_filesize, transcript)
                            )

    return pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_filesize", "transcript"]
    )


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
