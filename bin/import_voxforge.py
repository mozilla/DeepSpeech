#!/usr/bin/env python
import sys
import urllib2
import tarfile
import pandas
import re
import unicodedata
from glob import glob
from os import makedirs, path
from bs4 import BeautifulSoup
from tensorflow.python.platform import gfile
from tensorflow.contrib.learn.python.learn.datasets import base

def _download_and_preprocess_data(data_dir):
    # Conditionally download data to data_dir
    if not path.isdir(data_dir):
        makedirs(data_dir)
    archive_dir = data_dir+"/archive"
    if not path.isdir(archive_dir):
        makedirs(archive_dir)

    print("Downloading Voxforge data set into {} if not already present...".format(archive_dir))

    voxforge_url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit'
    html_page = urllib.request.urlopen(voxforge_url)
    soup = BeautifulSoup(html_page, 'html.parser')
    # list all links
    links = soup.find_all('a')
    refs = [l['href'] for l in links if ".tgz" in l['href']]
    def filename_of(x): return path.split(x)[1]

    for i, ref in enumerate(refs):
        print('Downloading {} / {} files'.format(i+1, len(refs)))
        download_url = voxforge_url + '/' + ref
        base.maybe_download(filename_of(download_url), archive_dir, download_url)

    # Conditionally extract data to dataset_dir
    if not path.isdir(path.join(data_dir,"test")):
        makedirs(path.join(data_dir,"test"))
    if not path.isdir(path.join(data_dir,"dev")):
        makedirs(path.join(data_dir,"dev"))
    if not path.isdir(path.join(data_dir,"train")):
        makedirs(path.join(data_dir,"train"))

    tarfiles = glob(path.join(archive_dir, "*.tgz"))
    number_of_files = len(tarfiles)
    number_of_test = number_of_files/100
    number_of_dev = number_of_files/100

    print("Extracting Voxforge data set into {} if not already present...".format(data_dir))
    for i,archive in enumerate(tarfiles):
        if i<number_of_test:
            dataset_dir = path.join(data_dir,"test")
        elif i<number_of_test+number_of_dev:
            dataset_dir = path.join(data_dir,"dev")
        else:
            dataset_dir = path.join(data_dir,"train")
        if not gfile.Exists(path.join(dataset_dir, '.'.join(filename_of(archive).split(".")[:-1]))):
            tar = tarfile.open(archive)
            tar.extractall(dataset_dir)
            tar.close()
            print('Extracting {} / {} files'.format(i+1, len(tarfiles)))

    # Generate data set
    print("Generating Voxforge data set into {}".format(data_dir))
    test_files = _generate_dataset(data_dir, "test")
    dev_files = _generate_dataset(data_dir, "dev")
    train_files = _generate_dataset(data_dir, "train")

    # Write sets to disk as CSV files
    train_files.to_csv(path.join(data_dir, "voxforge-train.csv"), index=False)
    dev_files.to_csv(path.join(data_dir, "voxforge-dev.csv"), index=False)
    test_files.to_csv(path.join(data_dir, "voxforge-test.csv"), index=False)


def _generate_dataset(data_dir, data_set):
    extracted_dir = path.join(data_dir, data_set)
    files = []
    for promts_file in glob(path.join(extracted_dir+"/*/etc/", "PROMPTS")):
        if path.isdir(path.join(promts_file[:-11],"wav")):
            with open(promts_file) as f:
                for line in f:
                    id = line.split(' ')[0].split('/')[-1]
                    sentence = ' '.join(line.split(' ')[1:])
                    sentence = re.sub("[^a-z']"," ",sentence.strip().lower())
                    transcript = ""
                    for token in sentence.split(" "):
                        word = token.strip()
                        if word!="" and word!=" ":
                            transcript += word + " "
                    transcript = unicodedata.normalize("NFKD", unicode(transcript.strip()))  \
                                              .encode("ascii", "ignore")                    \
                                              .decode("ascii", "ignore")
                    wav_file = path.join(promts_file[:-11],"wav/" + id + ".wav")
                    if gfile.Exists(wav_file):
                        wav_filesize = path.getsize(wav_file)
                        # remove audios that are shorter than 0.5s and longer than 20s.
                        # remove audios that are too short for transcript.
                        if (wav_filesize/32000)>0.5 and (wav_filesize/32000)<20 and transcript!="" and \
                            wav_filesize/len(transcript)>1400:
                            files.append((path.abspath(wav_file), wav_filesize, transcript))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

if __name__=="__main__":
    _download_and_preprocess_data(sys.argv[1])




