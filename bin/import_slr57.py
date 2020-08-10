#!/usr/bin/env python3
import csv
import os
import subprocess
import tarfile
import unicodedata
from glob import glob
from multiprocessing import Pool

import progressbar

from mozilla_voice_stt_training.util.downloader import SIMPLE_BAR, maybe_download
from mozilla_voice_stt_training.util.importers import (
    get_counter,
    get_imported_samples,
    get_importers_parser,
    get_validate_label,
    print_import_report,
)
from mvs_ctcdecoder import Alphabet

FIELDNAMES = ["wav_filename", "wav_filesize", "transcript"]
SAMPLE_RATE = 16000
MAX_SECS = 15

ARCHIVE_DIR_NAME = "African_Accented_French"
ARCHIVE_NAME = "African_Accented_French.tar.gz"
ARCHIVE_URL = "http://www.openslr.org/resources/57/" + ARCHIVE_NAME


def _download_and_preprocess_data(target_dir):
    # Making path absolute
    target_dir = os.path.abspath(target_dir)
    # Conditionally download data
    archive_path = maybe_download(ARCHIVE_NAME, target_dir, ARCHIVE_URL)
    # Conditionally extract data
    _maybe_extract(target_dir, ARCHIVE_DIR_NAME, archive_path)
    # Produce CSV files
    _maybe_convert_sets(target_dir, ARCHIVE_DIR_NAME)


def _maybe_extract(target_dir, extracted_data, archive_path):
    # If target_dir/extracted_data does not exist, extract archive in target_dir
    extracted_path = os.path.join(target_dir, extracted_data)
    if not os.path.exists(extracted_path):
        print('No directory "%s" - extracting archive...' % extracted_path)
        if not os.path.isdir(extracted_path):
            os.mkdir(extracted_path)
        tar = tarfile.open(archive_path)
        tar.extractall(target_dir)
        tar.close()
    else:
        print('Found directory "%s" - not extracting it from archive.' % archive_path)


def one_sample(sample):
    """ Take a audio file, and optionally convert it to 16kHz WAV """
    wav_filename = sample[0]
    file_size = -1
    frames = 0
    if os.path.exists(wav_filename):
        file_size = os.path.getsize(wav_filename)
        frames = int(
            subprocess.check_output(
                ["soxi", "-s", wav_filename], stderr=subprocess.STDOUT
            )
        )
    label = label_filter(sample[1])
    counter = get_counter()
    rows = []
    if file_size == -1:
        # Excluding samples that failed upon conversion
        counter["failed"] += 1
    elif label is None:
        # Excluding samples that failed on label validation
        counter["invalid_label"] += 1
    elif int(frames / SAMPLE_RATE * 1000 / 15 / 2) < len(str(label)):
        # Excluding samples that are too short to fit the transcript
        counter["too_short"] += 1
    elif frames / SAMPLE_RATE > MAX_SECS:
        # Excluding very long samples to keep a reasonable batch-size
        counter["too_long"] += 1
    else:
        # This one is good - keep it for the target CSV
        rows.append((wav_filename, file_size, label))
        counter["imported_time"] += frames
    counter["all"] += 1
    counter["total_time"] += frames

    return (counter, rows)


def _maybe_convert_sets(target_dir, extracted_data):
    extracted_dir = os.path.join(target_dir, extracted_data)
    # override existing CSV with normalized one
    target_csv_template = os.path.join(
        target_dir, ARCHIVE_DIR_NAME, ARCHIVE_NAME.replace(".tar.gz", "_{}.csv")
    )
    if os.path.isfile(target_csv_template):
        return

    wav_root_dir = os.path.join(extracted_dir)

    all_files = [
        "transcripts/train/yaounde/fn_text.txt",
        "transcripts/train/ca16_conv/transcripts.txt",
        "transcripts/train/ca16_read/conditioned.txt",
        "transcripts/dev/niger_west_african_fr/transcripts.txt",
        "speech/dev/niger_west_african_fr/niger_wav_file_name_transcript.tsv",
        "transcripts/devtest/ca16_read/conditioned.txt",
        "transcripts/test/ca16/prompts.txt",
    ]

    transcripts = {}
    for tr in all_files:
        with open(os.path.join(target_dir, ARCHIVE_DIR_NAME, tr), "r") as tr_source:
            for line in tr_source.readlines():
                line = line.strip()

                if ".tsv" in tr:
                    sep = "	"
                else:
                    sep = " "

                audio = os.path.basename(line.split(sep)[0])

                if not (".wav" in audio):
                    if ".tdf" in audio:
                        audio = audio.replace(".tdf", ".wav")
                    else:
                        audio += ".wav"

                transcript = " ".join(line.split(sep)[1:])
                transcripts[audio] = transcript

    # Get audiofile path and transcript for each sentence in tsv
    samples = []
    glob_dir = os.path.join(wav_root_dir, "**/*.wav")
    for record in glob(glob_dir, recursive=True):
        record_file = os.path.basename(record)
        if record_file in transcripts:
            samples.append((record, transcripts[record_file]))

    # Keep track of how many samples are good vs. problematic
    counter = get_counter()
    num_samples = len(samples)
    rows = []

    print("Importing WAV files...")
    pool = Pool()
    bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    for i, processed in enumerate(pool.imap_unordered(one_sample, samples), start=1):
        counter += processed[0]
        rows += processed[1]
        bar.update(i)
    bar.update(num_samples)
    pool.close()
    pool.join()

    with open(target_csv_template.format("train"), "w", encoding="utf-8", newline="") as train_csv_file:  # 80%
        with open(target_csv_template.format("dev"), "w", encoding="utf-8", newline="") as dev_csv_file:  # 10%
            with open(target_csv_template.format("test"), "w", encoding="utf-8", newline="") as test_csv_file:  # 10%
                train_writer = csv.DictWriter(train_csv_file, fieldnames=FIELDNAMES)
                train_writer.writeheader()
                dev_writer = csv.DictWriter(dev_csv_file, fieldnames=FIELDNAMES)
                dev_writer.writeheader()
                test_writer = csv.DictWriter(test_csv_file, fieldnames=FIELDNAMES)
                test_writer.writeheader()

                for i, item in enumerate(rows):
                    transcript = validate_label(item[2])
                    if not transcript:
                        continue
                    wav_filename = item[0]
                    i_mod = i % 10
                    if i_mod == 0:
                        writer = test_writer
                    elif i_mod == 1:
                        writer = dev_writer
                    else:
                        writer = train_writer
                    writer.writerow(
                        dict(
                            wav_filename=wav_filename,
                            wav_filesize=os.path.getsize(wav_filename),
                            transcript=transcript,
                        )
                    )

    imported_samples = get_imported_samples(counter)
    assert counter["all"] == num_samples
    assert len(rows) == imported_samples

    print_import_report(counter, SAMPLE_RATE, MAX_SECS)


def handle_args():
    parser = get_importers_parser(
        description="Importer for African Accented French dataset. More information on http://www.openslr.org/57/."
    )
    parser.add_argument(dest="target_dir")
    parser.add_argument(
        "--filter_alphabet",
        help="Exclude samples with characters not in provided alphabet",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Converts diacritic characters to their base ones",
    )
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    ALPHABET = Alphabet(CLI_ARGS.filter_alphabet) if CLI_ARGS.filter_alphabet else None
    validate_label = get_validate_label(CLI_ARGS)

    def label_filter(label):
        if CLI_ARGS.normalize:
            label = (
                unicodedata.normalize("NFKD", label.strip())
                .encode("ascii", "ignore")
                .decode("ascii", "ignore")
            )
        label = validate_label(label)
        if ALPHABET and label and not ALPHABET.CanEncode(label):
            label = None
        return label

    _download_and_preprocess_data(target_dir=CLI_ARGS.target_dir)
