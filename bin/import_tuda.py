#!/usr/bin/env python
"""
Downloads and prepares (parts of) the "German Distant Speech" corpus (TUDA) for DeepSpeech.py
Use "python3 import_tuda.py -h" for help
"""
import argparse
import csv
import os
import tarfile
import unicodedata
import wave
import xml.etree.ElementTree as ET
from collections import Counter

import progressbar

from mozilla_voice_stt_training.util.downloader import SIMPLE_BAR, maybe_download
from mozilla_voice_stt_training.util.importers import validate_label_eng as validate_label
from mvs_ctcdecoder import Alphabet

TUDA_VERSION = "v2"
TUDA_PACKAGE = "german-speechdata-package-{}".format(TUDA_VERSION)
TUDA_URL = "http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/{}.tar.gz".format(
    TUDA_PACKAGE
)
TUDA_ARCHIVE = "{}.tar.gz".format(TUDA_PACKAGE)

CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000

FIELDNAMES = ["wav_filename", "wav_filesize", "transcript"]


def maybe_extract(archive):
    extracted = os.path.join(CLI_ARGS.base_dir, TUDA_PACKAGE)
    if os.path.isdir(extracted):
        print('Found directory "{}" - not extracting.'.format(extracted))
    else:
        print('Extracting "{}"...'.format(archive))
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            bar = progressbar.ProgressBar(max_value=len(members), widgets=SIMPLE_BAR)
            for member in bar(members):
                tar.extract(member=member, path=CLI_ARGS.base_dir)
    return extracted


def in_alphabet(c):
    return ALPHABET.CanEncode(c) if ALPHABET else True


def check_and_prepare_sentence(sentence):
    sentence = sentence.lower().replace("co2", "c o zwei")
    chars = []
    for c in sentence:
        if CLI_ARGS.normalize and c not in "äöüß" and not in_alphabet(c):
            c = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("ascii", "ignore")
        for sc in c:
            if not in_alphabet(c):
                return None
            chars.append(sc)
    return validate_label("".join(chars))


def check_wav_file(wav_path, sentence):  # pylint: disable=too-many-return-statements
    try:
        with wave.open(wav_path, "r") as src_wav_file:
            rate = src_wav_file.getframerate()
            channels = src_wav_file.getnchannels()
            sample_width = src_wav_file.getsampwidth()
            milliseconds = int(src_wav_file.getnframes() * 1000 / rate)
        if rate != SAMPLE_RATE:
            return False, "wrong sample rate"
        if channels != CHANNELS:
            return False, "wrong number of channels"
        if sample_width != SAMPLE_WIDTH:
            return False, "wrong sample width"
        if milliseconds / len(sentence) < 30:
            return False, "too short"
        if milliseconds > CLI_ARGS.max_duration > 0:
            return False, "too long"
    except wave.Error:
        return False, "invalid wav file"
    except EOFError:
        return False, "premature EOF"
    return True, "OK"


def write_csvs(extracted):
    sample_counter = 0
    reasons = Counter()
    for sub_set in ["train", "dev", "test"]:
        set_path = os.path.join(extracted, sub_set)
        set_files = os.listdir(set_path)
        recordings = {}
        for file in set_files:
            if file.endswith(".xml"):
                recordings[file[:-4]] = []
        for file in set_files:
            if file.endswith(".wav") and "_" in file:
                prefix = file.split("_")[0]
                if prefix in recordings:
                    recordings[prefix].append(file)
        recordings = recordings.items()
        csv_path = os.path.join(
            CLI_ARGS.base_dir, "tuda-{}-{}.csv".format(TUDA_VERSION, sub_set)
        )
        print('Writing "{}"...'.format(csv_path))
        with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            set_dir = os.path.join(extracted, sub_set)
            bar = progressbar.ProgressBar(max_value=len(recordings), widgets=SIMPLE_BAR)
            for prefix, wav_names in bar(recordings):
                xml_path = os.path.join(set_dir, prefix + ".xml")
                meta = ET.parse(xml_path).getroot()
                sentence = list(meta.iter("cleaned_sentence"))[0].text
                sentence = check_and_prepare_sentence(sentence)
                if sentence is None:
                    reasons['alphabet filter'] += 1
                    continue
                for wav_name in wav_names:
                    sample_counter += 1
                    wav_path = os.path.join(set_path, wav_name)
                    keep, reason = check_wav_file(wav_path, sentence)
                    if keep:
                        writer.writerow(
                            {
                                "wav_filename": os.path.relpath(
                                    wav_path, CLI_ARGS.base_dir
                                ),
                                "wav_filesize": os.path.getsize(wav_path),
                                "transcript": sentence.lower(),
                            }
                        )
                    else:
                        reasons[reason] += 1
    if len(reasons.keys()) > 0:
        print("Excluded samples:")
        for reason, n in reasons.most_common():
            print(' - "{}": {} ({:.2f}%)'.format(reason, n, n * 100 / sample_counter))


def cleanup(archive):
    if not CLI_ARGS.keep_archive:
        print('Removing archive "{}"...'.format(archive))
        os.remove(archive)


def download_and_prepare():
    archive = maybe_download(TUDA_ARCHIVE, CLI_ARGS.base_dir, TUDA_URL)
    extracted = maybe_extract(archive)
    write_csvs(extracted)
    cleanup(archive)


def handle_args():
    parser = argparse.ArgumentParser(description="Import German Distant Speech (TUDA)")
    parser.add_argument("base_dir", help="Directory containing all data")
    parser.add_argument(
        "--max_duration",
        type=int,
        default=10000,
        help="Maximum sample duration in milliseconds",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Converts diacritic characters to their base ones",
    )
    parser.add_argument(
        "--alphabet",
        help="Exclude samples with characters not in provided alphabet file",
    )
    parser.add_argument(
        "--keep_archive",
        type=bool,
        default=True,
        help="If downloaded archives should be kept",
    )
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    ALPHABET = Alphabet(CLI_ARGS.alphabet) if CLI_ARGS.alphabet else None
    download_and_prepare()
