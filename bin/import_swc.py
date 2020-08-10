#!/usr/bin/env python
"""
Downloads and prepares (parts of) the "Spoken Wikipedia Corpora" for DeepSpeech.py
Use "python3 import_swc.py -h" for help
"""

import argparse
import csv
import os
import random
import re
import shutil
import sys
import tarfile
import unicodedata
import wave
import xml.etree.ElementTree as ET
from collections import Counter
from glob import glob
from multiprocessing.pool import ThreadPool

import progressbar
import sox

from mozilla_voice_stt_training.util.downloader import SIMPLE_BAR, maybe_download
from mozilla_voice_stt_training.util.importers import validate_label_eng as validate_label
from mvs_ctcdecoder import Alphabet

SWC_URL = "https://www2.informatik.uni-hamburg.de/nats/pub/SWC/SWC_{language}.tar"
SWC_ARCHIVE = "SWC_{language}.tar"
LANGUAGES = ["dutch", "english", "german"]
FIELDNAMES = ["wav_filename", "wav_filesize", "transcript"]
FIELDNAMES_EXT = FIELDNAMES + ["article", "speaker"]
CHANNELS = 1
SAMPLE_RATE = 16000
UNKNOWN = "<unknown>"
AUDIO_PATTERN = "audio*.ogg"
WAV_NAME = "audio.wav"
ALIGNED_NAME = "aligned.swc"

SUBSTITUTIONS = {
    "german": [
        (re.compile(r"\$"), "dollar"),
        (re.compile(r"€"), "euro"),
        (re.compile(r"£"), "pfund"),
        (
            re.compile(r"ein tausend ([^\s]+) hundert ([^\s]+) er( |$)"),
            r"\1zehnhundert \2er ",
        ),
        (re.compile(r"ein tausend (acht|neun) hundert"), r"\1zehnhundert"),
        (
            re.compile(
                r"eins punkt null null null punkt null null null punkt null null null"
            ),
            "eine milliarde",
        ),
        (
            re.compile(
                r"punkt null null null punkt null null null punkt null null null"
            ),
            "milliarden",
        ),
        (re.compile(r"eins punkt null null null punkt null null null"), "eine million"),
        (re.compile(r"punkt null null null punkt null null null"), "millionen"),
        (re.compile(r"eins punkt null null null"), "ein tausend"),
        (re.compile(r"punkt null null null"), "tausend"),
        (re.compile(r"punkt null"), None),
    ]
}

DONT_NORMALIZE = {"german": "ÄÖÜäöüß"}

PRE_FILTER = str.maketrans(dict.fromkeys("/()[]{}<>:"))


class Sample:
    def __init__(self, wav_path, start, end, text, article, speaker, sub_set=None):
        self.wav_path = wav_path
        self.start = start
        self.end = end
        self.text = text
        self.article = article
        self.speaker = speaker
        self.sub_set = sub_set


def fail(message):
    print(message)
    sys.exit(1)


def group(lst, get_key):
    groups = {}
    for obj in lst:
        key = get_key(obj)
        if key in groups:
            groups[key].append(obj)
        else:
            groups[key] = [obj]
    return groups


def get_sample_size(population_size):
    margin_of_error = 0.01
    fraction_picking = 0.50
    z_score = 2.58  # Corresponds to confidence level 99%
    numerator = (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
        margin_of_error ** 2
    )
    sample_size = 0
    for train_size in range(population_size, 0, -1):
        denominator = 1 + (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
            margin_of_error ** 2 * train_size
        )
        sample_size = int(numerator / denominator)
        if 2 * sample_size + train_size <= population_size:
            break
    return sample_size


def maybe_download_language(language):
    lang_upper = language[0].upper() + language[1:]
    return maybe_download(
        SWC_ARCHIVE.format(language=lang_upper),
        CLI_ARGS.base_dir,
        SWC_URL.format(language=lang_upper),
    )


def maybe_extract(data_dir, extracted_data, archive):
    extracted = os.path.join(data_dir, extracted_data)
    if os.path.isdir(extracted):
        print('Found directory "{}" - not extracting.'.format(extracted))
    else:
        print('Extracting "{}"...'.format(archive))
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            bar = progressbar.ProgressBar(max_value=len(members), widgets=SIMPLE_BAR)
            for member in bar(members):
                tar.extract(member=member, path=extracted)
    return extracted


def ignored(node):
    if node is None:
        return False
    if node.tag == "ignored":
        return True
    return ignored(node.find(".."))


def read_token(token):
    texts, start, end = [], None, None
    notes = token.findall("n")
    if len(notes) > 0:
        for note in notes:
            attributes = note.attrib
            if start is None and "start" in attributes:
                start = int(attributes["start"])
            if "end" in attributes:
                token_end = int(attributes["end"])
                if end is None or token_end > end:
                    end = token_end
            if "pronunciation" in attributes:
                t = attributes["pronunciation"]
                texts.append(t)
    elif "text" in token.attrib:
        texts.append(token.attrib["text"])
    return start, end, " ".join(texts)


def in_alphabet(alphabet, c):
    return alphabet.CanEncode(c) if alphabet else True



ALPHABETS = {}


def get_alphabet(language):
    if language in ALPHABETS:
        return ALPHABETS[language]
    alphabet_path = getattr(CLI_ARGS, language + "_alphabet")
    alphabet = Alphabet(alphabet_path) if alphabet_path else None
    ALPHABETS[language] = alphabet
    return alphabet


def label_filter(label, language):
    label = label.translate(PRE_FILTER)
    label = validate_label(label)
    if label is None:
        return None, "validation"
    substitutions = SUBSTITUTIONS[language] if language in SUBSTITUTIONS else []
    for pattern, replacement in substitutions:
        if replacement is None:
            if pattern.match(label):
                return None, "substitution rule"
        else:
            label = pattern.sub(replacement, label)
    chars = []
    dont_normalize = DONT_NORMALIZE[language] if language in DONT_NORMALIZE else ""
    alphabet = get_alphabet(language)
    for c in label:
        if CLI_ARGS.normalize and c not in dont_normalize and not in_alphabet(alphabet, c):
            c = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("ascii", "ignore")
        for sc in c:
            if not in_alphabet(alphabet, sc):
                return None, "illegal character"
            chars.append(sc)
    label = "".join(chars)
    label = validate_label(label)
    return label, "validation" if label is None else None


def collect_samples(base_dir, language):
    roots = []
    for root, _, files in os.walk(base_dir):
        if ALIGNED_NAME in files and WAV_NAME in files:
            roots.append(root)
    samples = []
    reasons = Counter()

    def add_sample(
        p_wav_path, p_article, p_speaker, p_start, p_end, p_text, p_reason="complete"
    ):
        if p_start is not None and p_end is not None and p_text is not None:
            duration = p_end - p_start
            text, filter_reason = label_filter(p_text, language)
            skip = False
            if filter_reason is not None:
                skip = True
                p_reason = filter_reason
            elif CLI_ARGS.exclude_unknown_speakers and p_speaker == UNKNOWN:
                skip = True
                p_reason = "unknown speaker"
            elif CLI_ARGS.exclude_unknown_articles and p_article == UNKNOWN:
                skip = True
                p_reason = "unknown article"
            elif duration > CLI_ARGS.max_duration > 0 and CLI_ARGS.ignore_too_long:
                skip = True
                p_reason = "exceeded duration"
            elif int(duration / 30) < len(text):
                skip = True
                p_reason = "too short to decode"
            elif duration / len(text) < 10:
                skip = True
                p_reason = "length duration ratio"
            if skip:
                reasons[p_reason] += 1
            else:
                samples.append(
                    Sample(p_wav_path, p_start, p_end, text, p_article, p_speaker)
                )
        elif p_start is None or p_end is None:
            reasons["missing timestamps"] += 1
        else:
            reasons["missing text"] += 1

    print("Collecting samples...")
    bar = progressbar.ProgressBar(max_value=len(roots), widgets=SIMPLE_BAR)
    for root in bar(roots):
        wav_path = os.path.join(root, WAV_NAME)
        aligned = ET.parse(os.path.join(root, ALIGNED_NAME))
        article = UNKNOWN
        speaker = UNKNOWN
        for prop in aligned.iter("prop"):
            attributes = prop.attrib
            if "key" in attributes and "value" in attributes:
                if attributes["key"] == "DC.identifier":
                    article = attributes["value"]
                elif attributes["key"] == "reader.name":
                    speaker = attributes["value"]
        for sentence in aligned.iter("s"):
            if ignored(sentence):
                continue
            split = False
            tokens = list(map(read_token, sentence.findall("t")))
            sample_start, sample_end, token_texts, sample_texts = None, None, [], []
            for token_start, token_end, token_text in tokens:
                if CLI_ARGS.exclude_numbers and any(c.isdigit() for c in token_text):
                    add_sample(
                        wav_path,
                        article,
                        speaker,
                        sample_start,
                        sample_end,
                        " ".join(sample_texts),
                        p_reason="has numbers",
                    )
                    sample_start, sample_end, token_texts, sample_texts = (
                        None,
                        None,
                        [],
                        [],
                    )
                    continue
                if sample_start is None:
                    sample_start = token_start
                if sample_start is None:
                    continue
                token_texts.append(token_text)
                if token_end is not None:
                    if (
                        token_start != sample_start
                        and token_end - sample_start > CLI_ARGS.max_duration > 0
                    ):
                        add_sample(
                            wav_path,
                            article,
                            speaker,
                            sample_start,
                            sample_end,
                            " ".join(sample_texts),
                            p_reason="split",
                        )
                        sample_start = sample_end
                        sample_texts = []
                        split = True
                    sample_end = token_end
                    sample_texts.extend(token_texts)
                    token_texts = []
            add_sample(
                wav_path,
                article,
                speaker,
                sample_start,
                sample_end,
                " ".join(sample_texts),
                p_reason="split" if split else "complete",
            )
    print("Skipped samples:")
    for reason, n in reasons.most_common():
        print(" - {}: {}".format(reason, n))
    return samples


def maybe_convert_one_to_wav(entry):
    root, _, files = entry
    transformer = sox.Transformer()
    transformer.convert(samplerate=SAMPLE_RATE, n_channels=CHANNELS)
    combiner = sox.Combiner()
    combiner.convert(samplerate=SAMPLE_RATE, n_channels=CHANNELS)
    output_wav = os.path.join(root, WAV_NAME)
    if os.path.isfile(output_wav):
        return
    files = sorted(glob(os.path.join(root, AUDIO_PATTERN)))
    try:
        if len(files) == 1:
            transformer.build(files[0], output_wav)
        elif len(files) > 1:
            wav_files = []
            for i, file in enumerate(files):
                wav_path = os.path.join(root, "audio{}.wav".format(i))
                transformer.build(file, wav_path)
                wav_files.append(wav_path)
            combiner.set_input_format(file_type=["wav"] * len(wav_files))
            combiner.build(wav_files, output_wav, "concatenate")
    except sox.core.SoxError:
        return


def maybe_convert_to_wav(base_dir):
    roots = list(os.walk(base_dir))
    print("Converting and joining source audio files...")
    bar = progressbar.ProgressBar(max_value=len(roots), widgets=SIMPLE_BAR)
    tp = ThreadPool()
    for _ in bar(tp.imap_unordered(maybe_convert_one_to_wav, roots)):
        pass
    tp.close()
    tp.join()


def assign_sub_sets(samples):
    sample_size = get_sample_size(len(samples))
    speakers = group(samples, lambda sample: sample.speaker).values()
    speakers = list(sorted(speakers, key=len))
    sample_sets = [[], []]
    while any(map(lambda s: len(s) < sample_size, sample_sets)) and len(speakers) > 0:
        for sample_set in sample_sets:
            if len(sample_set) < sample_size and len(speakers) > 0:
                sample_set.extend(speakers.pop(0))
    train_set = sum(speakers, [])
    if len(train_set) == 0:
        print(
            "WARNING: Unable to build dev and test sets without speaker bias as there is no speaker meta data"
        )
        random.seed(42)  # same source data == same output
        random.shuffle(samples)
        for index, sample in enumerate(samples):
            if index < sample_size:
                sample.sub_set = "dev"
            elif index < 2 * sample_size:
                sample.sub_set = "test"
            else:
                sample.sub_set = "train"
    else:
        for sub_set, sub_set_samples in [
            ("train", train_set),
            ("dev", sample_sets[0]),
            ("test", sample_sets[1]),
        ]:
            for sample in sub_set_samples:
                sample.sub_set = sub_set
    for sub_set, sub_set_samples in group(samples, lambda s: s.sub_set).items():
        t = sum(map(lambda s: s.end - s.start, sub_set_samples)) / (1000 * 60 * 60)
        print(
            'Sub-set "{}" with {} samples (duration: {:.2f} h)'.format(
                sub_set, len(sub_set_samples), t
            )
        )


def create_sample_dirs(language):
    print("Creating sample directories...")
    for set_name in ["train", "dev", "test"]:
        dir_path = os.path.join(CLI_ARGS.base_dir, language + "-" + set_name)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


def split_audio_files(samples, language):
    print("Splitting audio files...")
    sub_sets = Counter()
    src_wav_files = group(samples, lambda s: s.wav_path).items()
    bar = progressbar.ProgressBar(max_value=len(src_wav_files), widgets=SIMPLE_BAR)
    for wav_path, file_samples in bar(src_wav_files):
        file_samples = sorted(file_samples, key=lambda s: s.start)
        with wave.open(wav_path, "r") as src_wav_file:
            rate = src_wav_file.getframerate()
            for sample in file_samples:
                index = sub_sets[sample.sub_set]
                sample_wav_path = os.path.join(
                    CLI_ARGS.base_dir,
                    language + "-" + sample.sub_set,
                    "sample-{0:06d}.wav".format(index),
                )
                sample.wav_path = sample_wav_path
                sub_sets[sample.sub_set] += 1
                src_wav_file.setpos(int(sample.start * rate / 1000.0))
                data = src_wav_file.readframes(
                    int((sample.end - sample.start) * rate / 1000.0)
                )
                with wave.open(sample_wav_path, "w") as sample_wav_file:
                    sample_wav_file.setnchannels(src_wav_file.getnchannels())
                    sample_wav_file.setsampwidth(src_wav_file.getsampwidth())
                    sample_wav_file.setframerate(rate)
                    sample_wav_file.writeframes(data)


def write_csvs(samples, language):
    for sub_set, set_samples in group(samples, lambda s: s.sub_set).items():
        set_samples = sorted(set_samples, key=lambda s: s.wav_path)
        base_dir = os.path.abspath(CLI_ARGS.base_dir)
        csv_path = os.path.join(base_dir, language + "-" + sub_set + ".csv")
        print('Writing "{}"...'.format(csv_path))
        with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=FIELDNAMES_EXT if CLI_ARGS.add_meta else FIELDNAMES
            )
            writer.writeheader()
            bar = progressbar.ProgressBar(
                max_value=len(set_samples), widgets=SIMPLE_BAR
            )
            for sample in bar(set_samples):
                row = {
                    "wav_filename": os.path.relpath(sample.wav_path, base_dir),
                    "wav_filesize": os.path.getsize(sample.wav_path),
                    "transcript": sample.text,
                }
                if CLI_ARGS.add_meta:
                    row["article"] = sample.article
                    row["speaker"] = sample.speaker
                writer.writerow(row)


def cleanup(archive, language):
    if not CLI_ARGS.keep_archive:
        print('Removing archive "{}"...'.format(archive))
        os.remove(archive)
    language_dir = os.path.join(CLI_ARGS.base_dir, language)
    if not CLI_ARGS.keep_intermediate and os.path.isdir(language_dir):
        print('Removing intermediate files in "{}"...'.format(language_dir))
        shutil.rmtree(language_dir)


def prepare_language(language):
    archive = maybe_download_language(language)
    extracted = maybe_extract(CLI_ARGS.base_dir, language, archive)
    maybe_convert_to_wav(extracted)
    samples = collect_samples(extracted, language)
    assign_sub_sets(samples)
    create_sample_dirs(language)
    split_audio_files(samples, language)
    write_csvs(samples, language)
    cleanup(archive, language)


def handle_args():
    parser = argparse.ArgumentParser(description="Import Spoken Wikipedia Corpora")
    parser.add_argument("base_dir", help="Directory containing all data")
    parser.add_argument(
        "--language", default="all", help="One of (all|{})".format("|".join(LANGUAGES))
    )
    parser.add_argument(
        "--exclude_numbers",
        type=bool,
        default=True,
        help="If sequences with non-transliterated numbers should be excluded",
    )
    parser.add_argument(
        "--max_duration",
        type=int,
        default=10000,
        help="Maximum sample duration in milliseconds",
    )
    parser.add_argument(
        "--ignore_too_long",
        type=bool,
        default=False,
        help="If samples exceeding max_duration should be removed",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Converts diacritic characters to their base ones",
    )
    for language in LANGUAGES:
        parser.add_argument(
            "--{}_alphabet".format(language),
            help="Exclude {} samples with characters not in provided alphabet file".format(
                language
            ),
        )
    parser.add_argument(
        "--add_meta", action="store_true", help="Adds article and speaker CSV columns"
    )
    parser.add_argument(
        "--exclude_unknown_speakers",
        action="store_true",
        help="Exclude unknown speakers",
    )
    parser.add_argument(
        "--exclude_unknown_articles",
        action="store_true",
        help="Exclude unknown articles",
    )
    parser.add_argument(
        "--keep_archive",
        type=bool,
        default=True,
        help="If downloaded archives should be kept",
    )
    parser.add_argument(
        "--keep_intermediate",
        type=bool,
        default=False,
        help="If intermediate files should be kept",
    )
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    if CLI_ARGS.language == "all":
        for lang in LANGUAGES:
            prepare_language(lang)
    elif CLI_ARGS.language in LANGUAGES:
        prepare_language(CLI_ARGS.language)
    else:
        fail("Wrong language id")
