#!/usr/bin/env python
"""
Importer for dataset published from Centre de Conférence Pierre Mendès-France
Ministère de l'Économie, des Finances et de la Relance
"""

import csv
import sys
import os
import progressbar
import subprocess
import zipfile
from glob import glob
from multiprocessing import Pool

import hashlib
import decimal
import math
import unicodedata
import re
import sox
import xml.etree.ElementTree as ET

try:
    from num2words import num2words
except ImportError as ex:
    print("pip install num2words")
    sys.exit(1)

import requests
import json

from deepspeech_training.util.downloader import SIMPLE_BAR, maybe_download
from deepspeech_training.util.helpers import secs_to_hours
from deepspeech_training.util.importers import (
    get_counter,
    get_importers_parser,
    get_imported_samples,
    get_validate_label,
    print_import_report,
)
from ds_ctcdecoder import Alphabet

FIELDNAMES = ["wav_filename", "wav_filesize", "transcript"]
SAMPLE_RATE = 16000
CHANNELS = 1
BIT_DEPTH = 16
MAX_SECS = 10
MIN_SECS = 0.85

DATASET_RELEASE_CSV = "https://data.economie.gouv.fr/explore/dataset/transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B"
DATASET_RELEASE_SHA = [
    ("863d39a06a388c6491c6ff2f6450b151f38f1b57", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.001"),
    ("2f3a0305aa04c61220bb00b5a4e553e45dbf12e1", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.002"),
    ("5e55e9f1f844097349188ac875947e5a3d7fe9f1", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.003"),
    ("8bf54842cf07948ca5915e27a8bd5fa5139c06ae", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.004"),
    ("c8963504aadc015ac48f9af80058a0bb3440b94f", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.005"),
    ("d95e225e908621d83ce4e9795fd108d9d310e244", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.006"),
    ("de6ed9c2b0ee80ca879aae8ba7923cc93217d811", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.007"),
    ("234283c47dacfcd4450d836c52c25f3e807fc5f2", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.008"),
    ("4e6b67a688639bb72f8cd81782eaba604a8d32a6", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.009"),
    ("4165a51389777c8af8e6253d87bdacb877e8b3b0", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.010"),
    ("34322e7009780d97ef5bd02bf2f2c7a31f00baff", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.011"),
    ("48c5be3b2ca9d6108d525da6a03e91d93a95dbac", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.012"),
    ("87573172f506a189c2ebc633856fe11a2e9cd213", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.013"),
    ("6ab2c9e508e9278d5129f023e018725c4a7c69e8", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.014"),
    ("4f84df831ef46dce5d3ab3e21817687a2d8c12d0", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.015"),
    ("e69bfb079885c299cb81080ef88b1b8b57158aa6", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.016"),
    ("5f764ba788ee273981cf211b242c29b49ca22c5e", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.017"),
    ("b6aa81a959525363223494830c1e7307d4c4bae6", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.018"),
    ("91ddcf43c7bf113a6f2528b857c7ec22a50a148a", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.019"),
    ("fa1b29273dd77b9a7494983a2f9ae52654b931d7", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.020"),
    ("1113aef4f5e2be2f7fbf2d54b6c710c1c0e7135f", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.021"),
    ("ce6420d5d0b6b5135ba559f83e1a82d4d615c470", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.022"),
    ("d0976ed292ac24fcf1590d1ea195077c74b05471", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.023"),
    ("ec746cd6af066f62d9bf8d3b2f89174783ff4e3c", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.024"),
    ("570d9e1e84178e32fd867171d4b3aaecda1fd4fb", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.025"),
    ("c29ccc7467a75b2cae3d7f2e9fbbb2ab276cb8ac", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.026"),
    ("08406a51146d88e208704ce058c060a1e44efa50", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.027"),
    ("199aedad733a78ea1e7d47def9c71c6fd5795e02", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.028"),
    ("db856a068f92fb4f01f410bba42c7271de0f231a", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.029"),
    ("e3c0135f16c6c9d25a09dcb4f99a685438a84740", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.030"),
    ("e51b8bb9c0ae4339f98b4f21e6d29b825109f0ac", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.031"),
    ("be5e80cbc49b59b31ae33c30576ef0e1a162d84e", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.032"),
    ("501df58e3ff55fcfd75b93dab57566dc536948b8", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.033"),
    ("1a114875811a8cdcb8d85a9f6dbee78be3e05131", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.034"),
    ("465d824e7ee46448369182c0c28646d155a2249b", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.035"),
    ("37f341b1b266d143eb73138c31cfff3201b9d619", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.036"),
    ("9e7d8255987a8a77a90e0d4b55c8fd38b9fb5694", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.037"),
    ("54886755630cb080a53098cb1b6c951c6714a143", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.038"),
    ("4b7cbb0154697be795034f7a49712e882a97197a", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.039"),
    ("c8e1e565a0e7a1f6ff1dbfcefe677aa74a41d2f2", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip.040"),
]

def _download_and_preprocess_data(csv_url, target_dir):
    dataset_sources = os.path.join(target_dir, "transcriptionsXML_audioMP3_MEFR_CCPMF_2012-2020", "data.txt")
    if os.path.exists(dataset_sources):
        return dataset_sources

    # Making path absolute
    target_dir = os.path.abspath(target_dir)
    csv_ref = requests.get(csv_url).text.split('\r\n')[1:-1]
    for part in csv_ref:
        part_filename = requests.head(part).headers.get("Content-Disposition").split(" ")[1].split("=")[1].replace('"', "")
        if not os.path.exists(os.path.join(target_dir, part_filename)):
            part_path = maybe_download(part_filename, target_dir, part)

    def _big_sha1(fname):
        s = hashlib.sha1()
        buffer_size = 65536
        with open(fname, "rb") as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                s.update(data)
        return s.hexdigest()

    for (sha1, filename) in DATASET_RELEASE_SHA:
        print("Checking {} SHA1:".format(filename))
        csum = _big_sha1(os.path.join(target_dir, filename))
        if csum == sha1:
            print("\t{}: OK {}".format(filename, sha1))
        else:
            print("\t{}: ERROR: expected {}, computed {}".format(filename, sha1, csum))
        assert csum == sha1

    # Conditionally extract data
    _maybe_extract(target_dir, "transcriptionsXML_audioMP3_MEFR_CCPMF_2012-2020", "transcriptionsxml_audiomp3_mefr_ccpmf_2012-2020_2.zip", "transcriptionsXML_audioMP3_MEFR_CCPMF_2012-2020.zip")

    # Produce source text for extraction / conversion
    return _maybe_create_sources(os.path.join(target_dir, "transcriptionsXML_audioMP3_MEFR_CCPMF_2012-2020"))

def _maybe_extract(target_dir, extracted_data, archive, final):
    # If target_dir/extracted_data does not exist, extract archive in target_dir
    extracted_path = os.path.join(target_dir, extracted_data)
    archive_path = os.path.join(target_dir, archive)
    final_archive = os.path.join(extracted_path, final)

    if not os.path.exists(extracted_path):
        if not os.path.exists(archive_path):
            print('No archive "%s" - building ...' % archive_path)
            all_zip_parts = glob(archive_path + ".*")
            all_zip_parts.sort()
            cmdline = "cat {} > {}".format(" ".join(all_zip_parts), archive_path)
            print('Building with "%s"' % cmdline)
            subprocess.check_call(cmdline, shell=True, cwd=target_dir)
            assert os.path.exists(archive_path)

        print('No directory "%s" - extracting archive %s ...' % (extracted_path, archive_path))
        with zipfile.ZipFile(archive_path) as zip_f:
            zip_f.extractall(extracted_path)

        with zipfile.ZipFile(final_archive) as zip_f:
            zip_f.extractall(target_dir)
    else:
        print('Found directory "%s" - not extracting it from archive.' % extracted_path)

def _maybe_create_sources(dir):
    dataset_sources = os.path.join(dir, "data.txt")
    MP3 = glob(os.path.join(dir, "**", "*.mp3"))
    XML = glob(os.path.join(dir, "**", "*.xml"))

    MP3_XML_Scores = []
    MP3_XML_Fin = {}

    for f_mp3 in MP3:
        for f_xml in XML:
            b_mp3 = os.path.splitext(os.path.basename(f_mp3))[0]
            b_xml = os.path.splitext(os.path.basename(f_xml))[0]
            a_mp3 = b_mp3.split('_')
            a_xml = b_xml.split('_')
            score = 0
            date_mp3 = a_mp3[0]
            date_xml = a_xml[0]

            if date_mp3 != date_xml:
                continue

            for i in range(min(len(a_mp3), len(a_xml))):
                if (a_mp3[i] == a_xml[i]):
                    score += 1

            if score >= 1:
                MP3_XML_Scores.append((f_mp3, f_xml, score))

    # sort by score
    MP3_XML_Scores.sort(key=lambda x: x[2], reverse=True)
    for s_mp3, s_xml, score in MP3_XML_Scores:
        #print(s_mp3, s_xml, score)
        if score not in MP3_XML_Fin:
            MP3_XML_Fin[score] = {}

        if s_mp3 not in MP3_XML_Fin[score]:
            try:
                MP3.index(s_mp3)
                MP3.remove(s_mp3)
                MP3_XML_Fin[score][s_mp3] = s_xml
            except ValueError as ex:
                pass
        else:
            print("here:", MP3_XML_Fin[score][s_mp3], s_xml, file=sys.stderr)

    with open(dataset_sources, "w") as ds:
        for score in MP3_XML_Fin:
            for mp3 in MP3_XML_Fin[score]:
                xml = MP3_XML_Fin[score][mp3]
                if os.path.getsize(mp3) > 0 and os.path.getsize(xml) > 0:
                    mp3 = os.path.relpath(mp3, dir)
                    xml = os.path.relpath(xml, dir)
                    ds.write('{},{},{:0.2e}\n'.format(xml, mp3, 2.5e-4))
                else:
                    print("Empty file {} or {}".format(mp3, xml), file=sys.stderr)

    print("Missing XML pairs:", MP3, file=sys.stderr)
    return dataset_sources

def maybe_normalize_for_digits(label):
    # first, try to identify numbers like "50 000", "260 000"
    if " " in label:
        if any(s.isdigit() for s in label):
            thousands = re.compile(r"(\d{1,3}(?:\s*\d{3})*(?:,\d+)?)")
            maybe_thousands = thousands.findall(label)
            if len(maybe_thousands) > 0:
                while True:
                    (label, r) = re.subn(r"(\d)\s(\d{3})", "\\1\\2", label)
                    if r == 0:
                        break

    # this might be a time or duration in the form "hh:mm" or "hh:mm:ss"
    if ":" in label:
        for s in label.split(" "):
            if any(i.isdigit() for i in s):
                date_or_time = re.compile(r"(\d{1,2}):(\d{2}):?(\d{2})?")
                maybe_date_or_time = date_or_time.findall(s)
                if len(maybe_date_or_time) > 0:
                    maybe_hours   = maybe_date_or_time[0][0]
                    maybe_minutes = maybe_date_or_time[0][1]
                    maybe_seconds = maybe_date_or_time[0][2]
                    if len(maybe_seconds) > 0:
                        label = label.replace("{}:{}:{}".format(maybe_hours, maybe_minutes, maybe_seconds), "{} heures {} minutes et {} secondes".format(maybe_hours, maybe_minutes, maybe_seconds))
                    else:
                        label = label.replace("{}:{}".format(maybe_hours, maybe_minutes), "{} heures et {} minutes".format(maybe_hours, maybe_minutes))

    new_label = []
    # pylint: disable=too-many-nested-blocks
    for s in label.split(" "):
        if any(i.isdigit() for i in s):
            s = s.replace(",", ".") # num2words requires "." for floats
            s = s.replace("\"", "")  # clean some data, num2words would choke on 1959"

            last_c = s[-1]
            if not last_c.isdigit(): # num2words will choke on "0.6.", "24 ?"
                s = s[:-1]

            if any(i.isalpha() for i in s): # So we have any(isdigit()) **and** any(sialpha), like "3D"
                ns = []
                for c in s:
                    nc = c
                    if c.isdigit(): # convert "3" to "trois-"
                        try:
                            nc = num2words(c, lang="fr") + "-"
                        except decimal.InvalidOperation as ex:
                            print("decimal.InvalidOperation: '{}'".format(s))
                            raise ex
                    ns.append(nc)
                s = "".join(s)
            else:
                try:
                    s = num2words(s, lang="fr")
                except decimal.InvalidOperation as ex:
                    print("decimal.InvalidOperation: '{}'".format(s))
                    raise ex
        new_label.append(s)
    return " ".join(new_label)

def maybe_normalize_for_specials_chars(label):
    label = label.replace("%", "pourcents")
    label = label.replace("/", ", ") # clean intervals like 2019/2022 to "2019 2022"
    label = label.replace("-", ", ") # clean intervals like 70-80 to "70 80"
    label = label.replace("+", " plus ") # clean + and make it speakable
    label = label.replace("€", " euros ") # clean euro symbol and make it speakable
    label = label.replace("., ", ", ") # clean some strange "4.0., " (20181017_Innovation.xml)
    label = label.replace("°", " degré ") # clean some strange "°5" (20181210_EtatsGeneraux-1000_fre_750_und.xml)
    label = label.replace("...", ".") # remove ellipsis
    label = label.replace("..", ".") # remove broken ellipsis
    label = label.replace("m²", "mètre-carrés") # 20150616_Defi_Climat_3_wmv_0_fre_minefi.xml
    label = label.replace("[end]", "") # broken tag in 20150123_Entretiens_Tresor_PGM_wmv_0_fre_minefi.xml
    label = label.replace(u'\xB8c', " ç") # strange cedilla in 20150417_Printemps_Economie_2_wmv_0_fre_minefi.xml
    label = label.replace("C0²", "CO 2") # 20121016_Syteme_sante_copie_wmv_0_fre_minefi.xml
    return label

def maybe_normalize_for_anglicisms(label):
    label = label.replace("B2B", "B to B")
    label = label.replace("B2C", "B to C")
    label = label.replace("#", "hashtag ")
    label = label.replace("@", "at ")
    return label

def maybe_normalize(label):
    label = maybe_normalize_for_specials_chars(label)
    label = maybe_normalize_for_anglicisms(label)
    label = maybe_normalize_for_digits(label)
    return label

def one_sample(sample):
    file_size = -1
    frames = 0

    audio_source = sample[0]
    target_dir = sample[1]
    dataset_basename = sample[2]

    start_time = sample[3]
    duration = sample[4]
    label = label_filter_fun(sample[5])
    sample_id = sample[6]

    _wav_filename = os.path.basename(audio_source.replace(".wav", "_{:06}.wav".format(sample_id)))
    wav_fullname = os.path.join(target_dir, dataset_basename, _wav_filename)

    if not os.path.exists(wav_fullname):
        subprocess.check_output(["ffmpeg", "-i", audio_source, "-ss", str(start_time), "-t", str(duration), "-c", "copy", wav_fullname], stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    file_size = os.path.getsize(wav_fullname)
    frames = int(subprocess.check_output(["soxi", "-s", wav_fullname], stderr=subprocess.STDOUT))

    _counter = get_counter()
    _rows = []

    if file_size == -1:
        # Excluding samples that failed upon conversion
        _counter["failed"] += 1
    elif label is None:
        # Excluding samples that failed on label validation
        _counter["invalid_label"] += 1
    elif int(frames/SAMPLE_RATE*1000/10/2) < len(str(label)):
        # Excluding samples that are too short to fit the transcript
        _counter["too_short"] += 1
    elif frames/SAMPLE_RATE < MIN_SECS:
        # Excluding samples that are too short
        _counter["too_short"] += 1
    elif frames/SAMPLE_RATE > MAX_SECS:
        # Excluding very long samples to keep a reasonable batch-size
        _counter["too_long"] += 1
    else:
        # This one is good - keep it for the target CSV
        _rows.append((os.path.join(dataset_basename, _wav_filename), file_size, label))
        _counter["imported_time"] += frames
    _counter["all"] += 1
    _counter["total_time"] += frames

    return (_counter, _rows)

def _maybe_import_data(xml_file, audio_source, target_dir, rel_tol=1e-1):
    dataset_basename = os.path.splitext(os.path.split(xml_file)[1])[0]
    wav_root = os.path.join(target_dir, dataset_basename)
    if not os.path.exists(wav_root):
        os.makedirs(wav_root)

    source_frames = int(subprocess.check_output(["soxi", "-s", audio_source], stderr=subprocess.STDOUT))
    print("Source audio length: %s" % secs_to_hours(source_frames / SAMPLE_RATE))

    # Get audiofile path and transcript for each sentence in tsv
    samples = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    seq_id        = 0
    this_time     = 0.0
    this_duration = 0.0
    prev_time     = 0.0
    prev_duration = 0.0
    this_text     = ""
    for child in root:
        if child.tag == "row":
            cur_time     = float(child.attrib["timestamp"])
            cur_duration = float(child.attrib["timedur"])
            cur_text     = child.text

            if this_time == 0.0:
                this_time = cur_time

            delta    = cur_time - (prev_time + prev_duration)
            # rel_tol value is made from trial/error to try and compromise between:
            # - cutting enough to skip missing words
            # - not too short, not too long sentences
            is_close = math.isclose(cur_time, this_time + this_duration, rel_tol=rel_tol)
            is_short = ((this_duration + cur_duration + delta) < MAX_SECS)

            # when the previous element is close enough **and** this does not
            # go over MAX_SECS, we append content
            if (is_close and is_short):
                this_duration += cur_duration + delta
                this_text     += cur_text
            else:
                samples.append((audio_source, target_dir, dataset_basename, this_time, this_duration, this_text, seq_id))

                this_time     = cur_time
                this_duration = cur_duration
                this_text     = cur_text

                seq_id += 1

            prev_time     = cur_time
            prev_duration = cur_duration

    # Keep track of how many samples are good vs. problematic
    _counter = get_counter()
    num_samples = len(samples)
    _rows = []

    print("Processing XML data: {}".format(xml_file))
    pool = Pool()
    bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    for i, processed in enumerate(pool.imap_unordered(one_sample, samples), start=1):
        _counter += processed[0]
        _rows += processed[1]
        bar.update(i)
    bar.update(num_samples)
    pool.close()
    pool.join()

    imported_samples = get_imported_samples(_counter)
    assert _counter["all"] == num_samples
    assert len(_rows) == imported_samples

    print_import_report(_counter, SAMPLE_RATE, MAX_SECS)
    print("Import efficiency: %.1f%%" % ((_counter["total_time"] / source_frames)*100))
    print("")

    return _counter, _rows

def _maybe_convert_wav(mp3_filename, _wav_filename):
    if not os.path.exists(_wav_filename):
        print("Converting {} to WAV file: {}".format(mp3_filename, _wav_filename))
        transformer = sox.Transformer()
        transformer.convert(samplerate=SAMPLE_RATE, n_channels=CHANNELS, bitdepth=BIT_DEPTH)
        try:
            transformer.build(mp3_filename, _wav_filename)
        except sox.core.SoxError:
            pass

def write_general_csv(target_dir, _rows, _counter):
    target_csv_template = os.path.join(target_dir, "ccpmf_{}.csv")
    with open(target_csv_template.format("train"), "w") as train_csv_file:  # 80%
        with open(target_csv_template.format("dev"), "w") as dev_csv_file:  # 10%
            with open(target_csv_template.format("test"), "w") as test_csv_file:  # 10%
                train_writer = csv.DictWriter(train_csv_file, fieldnames=FIELDNAMES)
                train_writer.writeheader()
                dev_writer = csv.DictWriter(dev_csv_file, fieldnames=FIELDNAMES)
                dev_writer.writeheader()
                test_writer = csv.DictWriter(test_csv_file, fieldnames=FIELDNAMES)
                test_writer.writeheader()

                bar = progressbar.ProgressBar(max_value=len(_rows), widgets=SIMPLE_BAR)
                for i, item in enumerate(bar(_rows)):
                    i_mod = i % 10
                    if i_mod == 0:
                        writer = test_writer
                    elif i_mod == 1:
                        writer = dev_writer
                    else:
                        writer = train_writer
                    writer.writerow({"wav_filename": item[0], "wav_filesize": item[1], "transcript": item[2]})

    print("")
    print("~~~~ FINAL STATISTICS ~~~~")
    print_import_report(_counter, SAMPLE_RATE, MAX_SECS)
    print("~~~~ (FINAL STATISTICS) ~~~~")
    print("")

if __name__ == "__main__":
    PARSER = get_importers_parser(description="Import XML from Conference Centre for Economics, France")
    PARSER.add_argument("target_dir", help="Destination directory")
    PARSER.add_argument("--filter_alphabet", help="Exclude samples with characters not in provided alphabet")
    PARSER.add_argument("--normalize", action="store_true", help="Converts diacritic characters to their base ones")

    PARAMS = PARSER.parse_args()
    validate_label = get_validate_label(PARAMS)
    ALPHABET = Alphabet(PARAMS.filter_alphabet) if PARAMS.filter_alphabet else None

    def label_filter_fun(label):
        if PARAMS.normalize:
            label = unicodedata.normalize("NFKD", label.strip()) \
                .encode("ascii", "ignore") \
                .decode("ascii", "ignore")
        label = maybe_normalize(label)
        label = validate_label(label)
        if ALPHABET and label:
            try:
                ALPHABET.encode(label)
            except KeyError:
                label = None
        return label

    dataset_sources = _download_and_preprocess_data(csv_url=DATASET_RELEASE_CSV, target_dir=PARAMS.target_dir)
    sources_root_dir = os.path.dirname(dataset_sources)
    all_counter = get_counter()
    all_rows = []
    with open(dataset_sources, "r") as sources:
        for line in sources.readlines():
            d = line.split(",")
            this_xml = os.path.join(sources_root_dir, d[0])
            this_mp3 = os.path.join(sources_root_dir, d[1])
            this_rel = float(d[2])

            wav_filename = os.path.join(sources_root_dir, os.path.splitext(os.path.basename(this_mp3))[0] + ".wav")
            _maybe_convert_wav(this_mp3, wav_filename)
            counter, rows = _maybe_import_data(this_xml, wav_filename, sources_root_dir, this_rel)

            all_counter += counter
            all_rows += rows
    write_general_csv(sources_root_dir, _counter=all_counter, _rows=all_rows)
