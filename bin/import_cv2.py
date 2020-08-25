#!/usr/bin/env python
"""
Broadly speaking, this script takes the audio downloaded from Common Voice
for a certain language, in addition to the *.tsv files output by CorporaCreator,
and the script formats the data and transcripts to be in a state usable by
DeepSpeech.py
Use "python3 import_cv2.py -h" for help
"""
import csv
import os
import subprocess
import unicodedata
from multiprocessing import Pool

import progressbar
import sox

from mozilla_voice_stt_training.util.downloader import SIMPLE_BAR
from mozilla_voice_stt_training.util.importers import (
    get_counter,
    get_imported_samples,
    get_importers_parser,
    get_validate_label,
    print_import_report,
)
from ds_ctcdecoder import Alphabet

FIELDNAMES = ["wav_filename", "wav_filesize", "transcript"]
SAMPLE_RATE = 16000
MAX_SECS = 10
PARAMS = None
FILTER_OBJ = None


class LabelFilter:
    def __init__(self, normalize, alphabet, validate_fun):
        self.normalize = normalize
        self.alphabet = alphabet
        self.validate_fun = validate_fun

    def filter(self, label):
        if self.normalize:
            label = unicodedata.normalize("NFKD", label.strip()).encode("ascii", "ignore").decode("ascii", "ignore")
        label = self.validate_fun(label)
        if self.alphabet and label and not self.alphabet.CanEncode(label):
            label = None
        return label


def init_worker(params):
    global FILTER_OBJ  # pylint: disable=global-statement
    validate_label = get_validate_label(params)
    alphabet = Alphabet(params.filter_alphabet) if params.filter_alphabet else None
    FILTER_OBJ = LabelFilter(params.normalize, alphabet, validate_label)


def one_sample(sample):
    """ Take an audio file, and optionally convert it to 16kHz WAV """
    mp3_filename = sample[0]
    if not os.path.splitext(mp3_filename.lower())[1] == ".mp3":
        mp3_filename += ".mp3"
    # Storing wav files next to the mp3 ones - just with a different suffix
    wav_filename = os.path.splitext(mp3_filename)[0] + ".wav"
    _maybe_convert_wav(mp3_filename, wav_filename)
    file_size = -1
    frames = 0
    if os.path.exists(wav_filename):
        file_size = os.path.getsize(wav_filename)
        frames = int(
            subprocess.check_output(
                ["soxi", "-s", wav_filename], stderr=subprocess.STDOUT
            )
        )
    label = FILTER_OBJ.filter(sample[1])
    rows = []
    counter = get_counter()
    if file_size == -1:
        # Excluding samples that failed upon conversion
        counter["failed"] += 1
    elif label is None:
        # Excluding samples that failed on label validation
        counter["invalid_label"] += 1
    elif int(frames / SAMPLE_RATE * 1000 / 10 / 2) < len(str(label)):
        # Excluding samples that are too short to fit the transcript
        counter["too_short"] += 1
    elif frames / SAMPLE_RATE > MAX_SECS:
        # Excluding very long samples to keep a reasonable batch-size
        counter["too_long"] += 1
    else:
        # This one is good - keep it for the target CSV
        rows.append((os.path.split(wav_filename)[-1], file_size, label, sample[2]))
        counter["imported_time"] += frames
    counter["all"] += 1
    counter["total_time"] += frames

    return (counter, rows)


def _maybe_convert_set(dataset, tsv_dir, audio_dir, filter_obj, space_after_every_character=None, rows=None, exclude=None):
    exclude_transcripts = set()
    exclude_speakers = set()
    if exclude is not None:
        for sample in exclude:
            exclude_transcripts.add(sample[2])
            exclude_speakers.add(sample[3])

    if rows is None:
        rows = []
        input_tsv = os.path.join(os.path.abspath(tsv_dir), dataset + ".tsv")
        if not os.path.isfile(input_tsv):
            return rows
        print("Loading TSV file: ", input_tsv)
        # Get audiofile path and transcript for each sentence in tsv
        samples = []
        with open(input_tsv, encoding="utf-8") as input_tsv_file:
            reader = csv.DictReader(input_tsv_file, delimiter="\t")
            for row in reader:
                samples.append((os.path.join(audio_dir, row["path"]), row["sentence"], row["client_id"]))

        counter = get_counter()
        num_samples = len(samples)

        print("Importing mp3 files...")
        pool = Pool(initializer=init_worker, initargs=(PARAMS,))
        bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
        for i, processed in enumerate(pool.imap_unordered(one_sample, samples), start=1):
            counter += processed[0]
            rows += processed[1]
            bar.update(i)
        bar.update(num_samples)
        pool.close()
        pool.join()

        imported_samples = get_imported_samples(counter)
        assert counter["all"] == num_samples
        assert len(rows) == imported_samples
        print_import_report(counter, SAMPLE_RATE, MAX_SECS)

    output_csv = os.path.join(os.path.abspath(audio_dir), dataset + ".csv")
    print("Saving new DeepSpeech-formatted CSV file to: ", output_csv)
    with open(output_csv, "w", encoding="utf-8", newline="") as output_csv_file:
        print("Writing CSV file for DeepSpeech.py as: ", output_csv)
        writer = csv.DictWriter(output_csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        bar = progressbar.ProgressBar(max_value=len(rows), widgets=SIMPLE_BAR)
        for filename, file_size, transcript, speaker in bar(rows):
            if transcript in exclude_transcripts or speaker in exclude_speakers:
                continue
            if space_after_every_character:
                writer.writerow(
                    {
                        "wav_filename": filename,
                        "wav_filesize": file_size,
                        "transcript": " ".join(transcript),
                    }
                )
            else:
                writer.writerow(
                    {
                        "wav_filename": filename,
                        "wav_filesize": file_size,
                        "transcript": transcript,
                    }
                )
    return rows


def _preprocess_data(tsv_dir, audio_dir, space_after_every_character=False):
    exclude = []
    for dataset in ["test", "dev", "train", "validated", "other"]:
        set_samples = _maybe_convert_set(dataset, tsv_dir, audio_dir, space_after_every_character)
        if dataset in ["test", "dev"]:
            exclude += set_samples
        if dataset == "validated":
            _maybe_convert_set("train-all", tsv_dir, audio_dir, space_after_every_character,
                               rows=set_samples, exclude=exclude)


def _maybe_convert_wav(mp3_filename, wav_filename):
    if not os.path.exists(wav_filename):
        transformer = sox.Transformer()
        transformer.convert(samplerate=SAMPLE_RATE)
        try:
            transformer.build(mp3_filename, wav_filename)
        except sox.core.SoxError:
            pass


def parse_args():
    parser = get_importers_parser(description="Import CommonVoice v2.0 corpora")
    parser.add_argument("tsv_dir", help="Directory containing tsv files")
    parser.add_argument(
        "--audio_dir",
        help='Directory containing the audio clips - defaults to "<tsv_dir>/clips"',
    )
    parser.add_argument(
        "--filter_alphabet",
        help="Exclude samples with characters not in provided alphabet",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Converts diacritic characters to their base ones",
    )
    parser.add_argument(
        "--space_after_every_character",
        action="store_true",
        help="To help transcript join by white space",
    )
    return parser.parse_args()


def main():
    audio_dir = PARAMS.audio_dir if PARAMS.audio_dir else os.path.join(PARAMS.tsv_dir, "clips")
    _preprocess_data(PARAMS.tsv_dir, audio_dir, PARAMS.space_after_every_character)


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
