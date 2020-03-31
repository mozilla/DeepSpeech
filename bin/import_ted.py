#!/usr/bin/env python
import sys
import tarfile
import unicodedata
import wave
from glob import glob
from os import makedirs, path, remove, rmdir

import pandas
from sox import Transformer
from tensorflow.python.platform import gfile

from deepspeech_training.util.downloader import maybe_download
from deepspeech_training.util.stm import parse_stm_file


def _download_and_preprocess_data(data_dir):
    # Conditionally download data
    TED_DATA = "TEDLIUM_release2.tar.gz"
    TED_DATA_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"
    local_file = maybe_download(TED_DATA, data_dir, TED_DATA_URL)

    # Conditionally extract TED data
    TED_DIR = "TEDLIUM_release2"
    _maybe_extract(data_dir, TED_DIR, local_file)

    # Conditionally convert TED sph data to wav
    _maybe_convert_wav(data_dir, TED_DIR)

    # Conditionally split TED wav and text data into sentences
    train_files, dev_files, test_files = _maybe_split_sentences(data_dir, TED_DIR)

    # Write sets to disk as CSV files
    train_files.to_csv(path.join(data_dir, "ted-train.csv"), index=False)
    dev_files.to_csv(path.join(data_dir, "ted-dev.csv"), index=False)
    test_files.to_csv(path.join(data_dir, "ted-test.csv"), index=False)


def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()


def _maybe_convert_wav(data_dir, extracted_data):
    # Create extracted_data dir
    extracted_dir = path.join(data_dir, extracted_data)

    # Conditionally convert dev sph to wav
    _maybe_convert_wav_dataset(extracted_dir, "dev")

    # Conditionally convert train sph to wav
    _maybe_convert_wav_dataset(extracted_dir, "train")

    # Conditionally convert test sph to wav
    _maybe_convert_wav_dataset(extracted_dir, "test")


def _maybe_convert_wav_dataset(extracted_dir, data_set):
    # Create source dir
    source_dir = path.join(extracted_dir, data_set, "sph")

    # Create target dir
    target_dir = path.join(extracted_dir, data_set, "wav")

    # Conditionally convert sph files to wav files
    if not gfile.Exists(target_dir):
        # Create target_dir
        makedirs(target_dir)

        # Loop over sph files in source_dir and convert each to wav
        for sph_file in glob(path.join(source_dir, "*.sph")):
            transformer = Transformer()
            wav_filename = path.splitext(path.basename(sph_file))[0] + ".wav"
            wav_file = path.join(target_dir, wav_filename)
            transformer.build(sph_file, wav_file)
            remove(sph_file)

        # Remove source_dir
        rmdir(source_dir)


def _maybe_split_sentences(data_dir, extracted_data):
    # Create extracted_data dir
    extracted_dir = path.join(data_dir, extracted_data)

    # Conditionally split dev wav
    dev_files = _maybe_split_dataset(extracted_dir, "dev")

    # Conditionally split train wav
    train_files = _maybe_split_dataset(extracted_dir, "train")

    # Conditionally split test wav
    test_files = _maybe_split_dataset(extracted_dir, "test")

    return train_files, dev_files, test_files


def _maybe_split_dataset(extracted_dir, data_set):
    # Create stm dir
    stm_dir = path.join(extracted_dir, data_set, "stm")

    # Create wav dir
    wav_dir = path.join(extracted_dir, data_set, "wav")

    files = []

    # Loop over stm files and split corresponding wav
    for stm_file in glob(path.join(stm_dir, "*.stm")):
        # Parse stm file
        stm_segments = parse_stm_file(stm_file)

        # Open wav corresponding to stm_file
        wav_filename = path.splitext(path.basename(stm_file))[0] + ".wav"
        wav_file = path.join(wav_dir, wav_filename)
        origAudio = wave.open(wav_file, "r")

        # Loop over stm_segments and split wav_file for each segment
        for stm_segment in stm_segments:
            # Create wav segment filename
            start_time = stm_segment.start_time
            stop_time = stm_segment.stop_time
            new_wav_filename = (
                path.splitext(path.basename(stm_file))[0]
                + "-"
                + str(start_time)
                + "-"
                + str(stop_time)
                + ".wav"
            )
            new_wav_file = path.join(wav_dir, new_wav_filename)

            # If the wav segment filename does not exist create it
            if not gfile.Exists(new_wav_file):
                _split_wav(origAudio, start_time, stop_time, new_wav_file)

            new_wav_filesize = path.getsize(new_wav_file)
            files.append(
                (path.abspath(new_wav_file), new_wav_filesize, stm_segment.transcript)
            )

        # Close origAudio
        origAudio.close()

    return pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_filesize", "transcript"]
    )


def _split_wav(origAudio, start_time, stop_time, new_wav_file):
    frameRate = origAudio.getframerate()
    origAudio.setpos(int(start_time * frameRate))
    chunkData = origAudio.readframes(int((stop_time - start_time) * frameRate))
    chunkAudio = wave.open(new_wav_file, "w")
    chunkAudio.setnchannels(origAudio.getnchannels())
    chunkAudio.setsampwidth(origAudio.getsampwidth())
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
