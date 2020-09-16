#!/usr/bin/env python
# ensure that you have downloaded the LDC dataset LDC97S62 and tar exists in a folder e.g.
# ./data/swb/swb1_LDC97S62.tgz
# from the deepspeech directory run with: ./bin/import_swb.py ./data/swb/
import codecs
import fnmatch
import os
import subprocess
import sys
import tarfile
import unicodedata
import wave

import librosa
import pandas
import requests
import soundfile  # <= Has an external dependency on libsndfile

from deepspeech_training.util.importers import validate_label_eng as validate_label

# ARCHIVE_NAME refers to ISIP alignments from 01/29/03
ARCHIVE_NAME = "switchboard_word_alignments.tar.gz"
ARCHIVE_URL = "http://www.openslr.org/resources/5/"
ARCHIVE_DIR_NAME = "LDC97S62"
LDC_DATASET = "swb1_LDC97S62.tgz"


def download_file(folder, url):
    # https://stackoverflow.com/a/16696317/738515
    local_filename = url.split("/")[-1]
    full_filename = os.path.join(folder, local_filename)
    r = requests.get(url, stream=True)
    with open(full_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return full_filename


def maybe_download(archive_url, target_dir, ldc_dataset):
    # If archive file does not exist, download it...
    archive_path = os.path.join(target_dir, ldc_dataset)
    ldc_path = archive_url + ldc_dataset
    if not os.path.exists(target_dir):
        print('No path "%s" - creating ...' % target_dir)
        os.makedirs(target_dir)

    if not os.path.exists(archive_path):
        print('No archive "%s" - downloading...' % archive_path)
        download_file(target_dir, ldc_path)
    else:
        print('Found archive "%s" - not downloading.' % archive_path)
    return archive_path


def _download_and_preprocess_data(data_dir):
    new_data_dir = os.path.join(data_dir, ARCHIVE_DIR_NAME)
    target_dir = os.path.abspath(new_data_dir)
    archive_path = os.path.abspath(os.path.join(data_dir, LDC_DATASET))

    # Check swb1_LDC97S62.tgz then extract
    assert os.path.isfile(archive_path)
    _extract(target_dir, archive_path)

    # Transcripts
    transcripts_path = maybe_download(ARCHIVE_URL, target_dir, ARCHIVE_NAME)
    _extract(target_dir, transcripts_path)

    # Check swb1_d1/2/3/4/swb_ms98_transcriptions
    expected_folders = [
        "swb1_d1",
        "swb1_d2",
        "swb1_d3",
        "swb1_d4",
        "swb_ms98_transcriptions",
    ]
    assert all([os.path.isdir(os.path.join(target_dir, e)) for e in expected_folders])

    # Conditionally convert swb sph data to wav
    _maybe_convert_wav(target_dir, "swb1_d1", "swb1_d1-wav")
    _maybe_convert_wav(target_dir, "swb1_d2", "swb1_d2-wav")
    _maybe_convert_wav(target_dir, "swb1_d3", "swb1_d3-wav")
    _maybe_convert_wav(target_dir, "swb1_d4", "swb1_d4-wav")

    # Conditionally split wav data
    d1 = _maybe_split_wav_and_sentences(
        target_dir, "swb_ms98_transcriptions", "swb1_d1-wav", "swb1_d1-split-wav"
    )
    d2 = _maybe_split_wav_and_sentences(
        target_dir, "swb_ms98_transcriptions", "swb1_d2-wav", "swb1_d2-split-wav"
    )
    d3 = _maybe_split_wav_and_sentences(
        target_dir, "swb_ms98_transcriptions", "swb1_d3-wav", "swb1_d3-split-wav"
    )
    d4 = _maybe_split_wav_and_sentences(
        target_dir, "swb_ms98_transcriptions", "swb1_d4-wav", "swb1_d4-split-wav"
    )

    swb_files = d1.append(d2).append(d3).append(d4)

    train_files, dev_files, test_files = _split_sets(swb_files)

    # Write sets to disk as CSV files
    train_files.to_csv(os.path.join(target_dir, "swb-train.csv"), index=False)
    dev_files.to_csv(os.path.join(target_dir, "swb-dev.csv"), index=False)
    test_files.to_csv(os.path.join(target_dir, "swb-test.csv"), index=False)


def _extract(target_dir, archive_path):
    with tarfile.open(archive_path) as tar:
        tar.extractall(target_dir)


def _maybe_convert_wav(data_dir, original_data, converted_data):
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)

    # Conditionally convert sph files to wav files
    if os.path.exists(target_dir):
        print("skipping maybe_convert_wav")
        return

    # Create target_dir
    os.makedirs(target_dir)

    # Loop over sph files in source_dir and convert each to 16-bit PCM wav
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.sph"):
            for channel in ["1", "2"]:
                sph_file = os.path.join(root, filename)
                wav_filename = (
                    os.path.splitext(os.path.basename(sph_file))[0]
                    + "-"
                    + channel
                    + ".wav"
                )
                wav_file = os.path.join(target_dir, wav_filename)
                temp_wav_filename = (
                    os.path.splitext(os.path.basename(sph_file))[0]
                    + "-"
                    + channel
                    + "-temp.wav"
                )
                temp_wav_file = os.path.join(target_dir, temp_wav_filename)
                print("converting {} to {}".format(sph_file, temp_wav_file))
                subprocess.check_call(
                    [
                        "sph2pipe",
                        "-c",
                        channel,
                        "-p",
                        "-f",
                        "rif",
                        sph_file,
                        temp_wav_file,
                    ]
                )
                print("upsampling {} to {}".format(temp_wav_file, wav_file))
                audioData, frameRate = librosa.load(temp_wav_file, sr=16000, mono=True)
                soundfile.write(wav_file, audioData, frameRate, "PCM_16")
                os.remove(temp_wav_file)


def _parse_transcriptions(trans_file):
    segments = []
    with codecs.open(trans_file, "r", "utf-8") as fin:
        for line in fin:
            if line.startswith("#") or len(line) <= 1:
                continue

            tokens = line.split()
            start_time = float(tokens[1])
            stop_time = float(tokens[2])
            transcript = validate_label(" ".join(tokens[3:]))

            if transcript == None:
                continue

            # We need to do the encode-decode dance here because encode
            # returns a bytes() object on Python 3, and text_to_char_array
            # expects a string.
            transcript = (
                unicodedata.normalize("NFKD", transcript)
                .encode("ascii", "ignore")
                .decode("ascii", "ignore")
            )

            segments.append(
                {
                    "start_time": start_time,
                    "stop_time": stop_time,
                    "transcript": transcript,
                }
            )
    return segments


def _maybe_split_wav_and_sentences(data_dir, trans_data, original_data, converted_data):
    trans_dir = os.path.join(data_dir, trans_data)
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)
    if os.path.exists(target_dir):
        print("skipping maybe_split_wav")
        return

    os.makedirs(target_dir)

    files = []

    # Loop over transcription files and split corresponding wav
    for root, dirnames, filenames in os.walk(trans_dir):
        for filename in fnmatch.filter(filenames, "*.text"):
            if "trans" not in filename:
                continue
            trans_file = os.path.join(root, filename)
            segments = _parse_transcriptions(trans_file)

            # Open wav corresponding to transcription file
            channel = ("2", "1")[
                (os.path.splitext(os.path.basename(trans_file))[0])[6] == "A"
            ]
            wav_filename = (
                "sw0"
                + (os.path.splitext(os.path.basename(trans_file))[0])[2:6]
                + "-"
                + channel
                + ".wav"
            )
            wav_file = os.path.join(source_dir, wav_filename)

            print("splitting {} according to {}".format(wav_file, trans_file))

            if not os.path.exists(wav_file):
                print("skipping. does not exist:" + wav_file)
                continue

            origAudio = wave.open(wav_file, "r")

            # Loop over segments and split wav_file for each segment
            for segment in segments:
                # Create wav segment filename
                start_time = segment["start_time"]
                stop_time = segment["stop_time"]
                new_wav_filename = (
                    os.path.splitext(os.path.basename(trans_file))[0]
                    + "-"
                    + str(start_time)
                    + "-"
                    + str(stop_time)
                    + ".wav"
                )
                if _is_wav_too_short(new_wav_filename):
                    continue
                new_wav_file = os.path.join(target_dir, new_wav_filename)

                _split_wav(origAudio, start_time, stop_time, new_wav_file)

                new_wav_filesize = os.path.getsize(new_wav_file)
                transcript = segment["transcript"]
                files.append(
                    (os.path.abspath(new_wav_file), new_wav_filesize, transcript)
                )

            # Close origAudio
            origAudio.close()

    return pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_filesize", "transcript"]
    )


def _is_wav_too_short(wav_filename):
    short_wav_filenames = [
        "sw2986A-ms98-a-trans-80.6385-83.358875.wav",
        "sw2663A-ms98-a-trans-161.12025-164.213375.wav",
    ]
    return wav_filename in short_wav_filenames


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


def _split_sets(filelist):
    # We initially split the entire set into 80% train and 20% test, then
    # split the train set into 80% train and 20% validation.
    train_beg = 0
    train_end = int(0.8 * len(filelist))

    dev_beg = int(0.8 * train_end)
    dev_end = train_end
    train_end = dev_beg

    test_beg = dev_end
    test_end = len(filelist)

    return (
        filelist[train_beg:train_end],
        filelist[dev_beg:dev_end],
        filelist[test_beg:test_end],
    )


def _read_data_set(
    filelist,
    thread_count,
    batch_size,
    numcep,
    numcontext,
    stride=1,
    offset=0,
    next_index=lambda i: i + 1,
    limit=0,
):
    # Optionally apply dataset size limit
    if limit > 0:
        filelist = filelist.iloc[:limit]

    filelist = filelist[offset::stride]

    # Return DataSet
    return DataSet(
        txt_files, thread_count, batch_size, numcep, numcontext, next_index=next_index
    )


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
