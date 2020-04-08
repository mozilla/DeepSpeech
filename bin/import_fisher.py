#!/usr/bin/env python
import codecs
import fnmatch
import os
import subprocess
import sys
import unicodedata

import librosa
import pandas
import soundfile  # <= Has an external dependency on libsndfile

from deepspeech_training.util.importers import validate_label_eng as validate_label

# Prerequisite: Having the sph2pipe tool in your PATH:
# https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools


def _download_and_preprocess_data(data_dir):
    # Assume data_dir contains extracted LDC2004S13, LDC2004T19, LDC2005S13, LDC2005T19

    # Conditionally convert Fisher sph data to wav
    _maybe_convert_wav(data_dir, "LDC2004S13", "fisher-2004-wav")
    _maybe_convert_wav(data_dir, "LDC2005S13", "fisher-2005-wav")

    # Conditionally split Fisher wav data
    all_2004 = _split_wav_and_sentences(
        data_dir,
        original_data="fisher-2004-wav",
        converted_data="fisher-2004-split-wav",
        trans_data=os.path.join("LDC2004T19", "fe_03_p1_tran", "data", "trans"),
    )
    all_2005 = _split_wav_and_sentences(
        data_dir,
        original_data="fisher-2005-wav",
        converted_data="fisher-2005-split-wav",
        trans_data=os.path.join("LDC2005T19", "fe_03_p2_tran", "data", "trans"),
    )

    # The following files have incorrect transcripts that are much longer than
    # their audio source. The result is that we end up with more labels than time
    # slices, which breaks CTC.
    all_2004.loc[
        all_2004["wav_filename"].str.endswith("fe_03_00265-33.53-33.81.wav"),
        "transcript",
    ] = "correct"
    all_2004.loc[
        all_2004["wav_filename"].str.endswith("fe_03_00991-527.39-528.3.wav"),
        "transcript",
    ] = "that's one of those"
    all_2005.loc[
        all_2005["wav_filename"].str.endswith("fe_03_10282-344.42-344.84.wav"),
        "transcript",
    ] = "they don't want"
    all_2005.loc[
        all_2005["wav_filename"].str.endswith("fe_03_10677-101.04-106.41.wav"),
        "transcript",
    ] = "uh my mine yeah the german shepherd pitbull mix he snores almost as loud as i do"

    # The following file is just a short sound and not at all transcribed like provided.
    # So we just exclude it.
    all_2004 = all_2004[
        ~all_2004["wav_filename"].str.endswith("fe_03_00027-393.8-394.05.wav")
    ]

    # The following file is far too long and would ruin our training batch size.
    # So we just exclude it.
    all_2005 = all_2005[
        ~all_2005["wav_filename"].str.endswith("fe_03_11487-31.09-234.06.wav")
    ]

    # The following file is too large for its transcript, so we just exclude it.
    all_2004 = all_2004[
        ~all_2004["wav_filename"].str.endswith("fe_03_01326-307.42-307.93.wav")
    ]

    # Conditionally split Fisher data into train/validation/test sets
    train_2004, dev_2004, test_2004 = _split_sets(all_2004)
    train_2005, dev_2005, test_2005 = _split_sets(all_2005)

    # Join 2004 and 2005 data
    train_files = train_2004.append(train_2005)
    dev_files = dev_2004.append(dev_2005)
    test_files = test_2004.append(test_2005)

    # Write sets to disk as CSV files
    train_files.to_csv(os.path.join(data_dir, "fisher-train.csv"), index=False)
    dev_files.to_csv(os.path.join(data_dir, "fisher-dev.csv"), index=False)
    test_files.to_csv(os.path.join(data_dir, "fisher-test.csv"), index=False)


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
            sph_file = os.path.join(root, filename)
            for channel in ["1", "2"]:
                wav_filename = (
                    os.path.splitext(os.path.basename(sph_file))[0]
                    + "_c"
                    + channel
                    + ".wav"
                )
                wav_file = os.path.join(target_dir, wav_filename)
                print("converting {} to {}".format(sph_file, wav_file))
                subprocess.check_call(
                    ["sph2pipe", "-c", channel, "-p", "-f", "rif", sph_file, wav_file]
                )


def _parse_transcriptions(trans_file):
    segments = []
    with codecs.open(trans_file, "r", "utf-8") as fin:
        for line in fin:
            if line.startswith("#") or len(line) <= 1:
                continue

            tokens = line.split()
            start_time = float(tokens[0])
            stop_time = float(tokens[1])
            speaker = tokens[2]
            transcript = " ".join(tokens[3:])

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
                    "speaker": speaker,
                    "transcript": transcript,
                }
            )
    return segments


def _split_wav_and_sentences(data_dir, trans_data, original_data, converted_data):
    trans_dir = os.path.join(data_dir, trans_data)
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = []

    # Loop over transcription files and split corresponding wav
    for root, dirnames, filenames in os.walk(trans_dir):
        for filename in fnmatch.filter(filenames, "*.txt"):
            trans_file = os.path.join(root, filename)
            segments = _parse_transcriptions(trans_file)

            # Open wav corresponding to transcription file
            wav_filenames = [
                os.path.splitext(os.path.basename(trans_file))[0]
                + "_c"
                + channel
                + ".wav"
                for channel in ["1", "2"]
            ]
            wav_files = [
                os.path.join(source_dir, wav_filename) for wav_filename in wav_filenames
            ]

            print("splitting {} according to {}".format(wav_files, trans_file))

            origAudios = [
                librosa.load(wav_file, sr=16000, mono=False) for wav_file in wav_files
            ]

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
                new_wav_file = os.path.join(target_dir, new_wav_filename)

                channel = 0 if segment["speaker"] == "A:" else 1
                _split_and_resample_wav(
                    origAudios[channel], start_time, stop_time, new_wav_file
                )

                new_wav_filesize = os.path.getsize(new_wav_file)
                transcript = validate_label(segment["transcript"])
                if transcript != None:
                    files.append(
                        (os.path.abspath(new_wav_file), new_wav_filesize, transcript)
                    )

    return pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_filesize", "transcript"]
    )


def _split_audio(origAudio, start_time, stop_time):
    audioData, frameRate = origAudio
    nChannels = len(audioData.shape)
    startIndex = int(start_time * frameRate)
    stopIndex = int(stop_time * frameRate)
    return (
        audioData[startIndex:stopIndex]
        if 1 == nChannels
        else audioData[:, startIndex:stopIndex]
    )


def _split_and_resample_wav(origAudio, start_time, stop_time, new_wav_file):
    frameRate = origAudio[1]
    chunkData = _split_audio(origAudio, start_time, stop_time)
    soundfile.write(new_wav_file, chunkData, frameRate, "PCM_16")


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


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
