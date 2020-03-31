#!/usr/bin/env python

import csv
import logging
import math
import os
import subprocess
import urllib
from pathlib import Path

import pandas as pd
from sox import Transformer

import swifter
from deepspeech_training.util.importers import get_importers_parser, get_validate_label

__version__ = "0.1.0"
_logger = logging.getLogger(__name__)


MAX_SECS = 10
BITDEPTH = 16
N_CHANNELS = 1
SAMPLE_RATE = 16000

DEV_PERCENTAGE = 0.10
TRAIN_PERCENTAGE = 0.80


def parse_args(args):
    """Parse command line parameters
    Args:
      args ([str]): Command line parameters as list of strings
    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = get_importers_parser(description="Imports GramVaani data for Deep Speech")
    parser.add_argument(
        "--version",
        action="version",
        version="GramVaaniImporter {ver}".format(ver=__version__),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        required=False,
        help="set loglevel to INFO",
        dest="loglevel",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        action="store_const",
        required=False,
        help="set loglevel to DEBUG",
        dest="loglevel",
        const=logging.DEBUG,
    )
    parser.add_argument(
        "-c",
        "--csv_filename",
        required=True,
        help="Path to the GramVaani csv",
        dest="csv_filename",
    )
    parser.add_argument(
        "-t",
        "--target_dir",
        required=True,
        help="Directory in which to save the importer GramVaani data",
        dest="target_dir",
    )
    return parser.parse_args(args)


def setup_logging(level):
    """Setup basic logging
    Args:
      level (int): minimum log level for emitting messages
    """
    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=level, stream=sys.stdout, format=format, datefmt="%Y-%m-%d %H:%M:%S"
    )


class GramVaaniCSV:
    """GramVaaniCSV representing a GramVaani dataset.
    Args:
      csv_filename (str): Path to the GramVaani csv
    Attributes:
        data (:class:`pandas.DataFrame`): `pandas.DataFrame` Containing the GramVaani csv data
    """

    def __init__(self, csv_filename):
        self.data = self._parse_csv(csv_filename)

    def _parse_csv(self, csv_filename):
        _logger.info("Parsing csv file...%s", os.path.abspath(csv_filename))
        data = pd.read_csv(
            os.path.abspath(csv_filename),
            names=[
                "piece_id",
                "audio_url",
                "transcript_labelled",
                "transcript",
                "labels",
                "content_filename",
                "audio_length",
                "user_id",
            ],
            usecols=["audio_url", "transcript", "audio_length"],
            skiprows=[0],
            engine="python",
            encoding="utf-8",
            quotechar='"',
            quoting=csv.QUOTE_ALL,
        )
        data.dropna(inplace=True)
        _logger.info("Parsed %d lines csv file." % len(data))
        return data


class GramVaaniDownloader:
    """GramVaaniDownloader downloads a GramVaani dataset.
    Args:
      gram_vaani_csv (GramVaaniCSV): A GramVaaniCSV representing the data to download
      target_dir (str): The path to download the data to
    Attributes:
        data (:class:`pandas.DataFrame`): `pandas.DataFrame` Containing the GramVaani csv data
    """

    def __init__(self, gram_vaani_csv, target_dir):
        self.target_dir = target_dir
        self.data = gram_vaani_csv.data

    def download(self):
        """Downloads the data associated with this instance
        Return:
          mp3_directory (os.path): The directory into which the associated mp3's were downloaded
        """
        mp3_directory = self._pre_download()
        self.data.swifter.apply(
            func=lambda arg: self._download(*arg, mp3_directory), axis=1, raw=True
        )
        return mp3_directory

    def _pre_download(self):
        mp3_directory = os.path.join(self.target_dir, "mp3")
        if not os.path.exists(self.target_dir):
            _logger.info("Creating directory...%s", self.target_dir)
            os.mkdir(self.target_dir)
        if not os.path.exists(mp3_directory):
            _logger.info("Creating directory...%s", mp3_directory)
            os.mkdir(mp3_directory)
        return mp3_directory

    def _download(self, audio_url, transcript, audio_length, mp3_directory):
        if audio_url == "audio_url":
            return
        mp3_filename = os.path.join(mp3_directory, os.path.basename(audio_url))
        if not os.path.exists(mp3_filename):
            _logger.debug("Downloading mp3 file...%s", audio_url)
            urllib.request.urlretrieve(audio_url, mp3_filename)
        else:
            _logger.debug("Already downloaded mp3 file...%s", audio_url)


class GramVaaniConverter:
    """GramVaaniConverter converts the mp3's to wav's for a GramVaani dataset.
    Args:
      target_dir (str): The path to download the data from
      mp3_directory (os.path): The path containing the GramVaani mp3's
    Attributes:
        target_dir (str): The target directory passed as a command line argument
        mp3_directory (os.path): The path containing the GramVaani mp3's
    """

    def __init__(self, target_dir, mp3_directory):
        self.target_dir = target_dir
        self.mp3_directory = Path(mp3_directory)

    def convert(self):
        """Converts the mp3's associated with this instance to wav's
        Return:
          wav_directory (os.path): The directory into which the associated wav's were downloaded
        """
        wav_directory = self._pre_convert()
        for mp3_filename in self.mp3_directory.glob("**/*.mp3"):
            wav_filename = os.path.join(
                wav_directory,
                os.path.splitext(os.path.basename(mp3_filename))[0] + ".wav",
            )
            if not os.path.exists(wav_filename):
                _logger.debug(
                    "Converting mp3 file %s to wav file %s"
                    % (mp3_filename, wav_filename)
                )
                transformer = Transformer()
                transformer.convert(
                    samplerate=SAMPLE_RATE, n_channels=N_CHANNELS, bitdepth=BITDEPTH
                )
                transformer.build(str(mp3_filename), str(wav_filename))
            else:
                _logger.debug(
                    "Already converted mp3 file %s to wav file %s"
                    % (mp3_filename, wav_filename)
                )
        return wav_directory

    def _pre_convert(self):
        wav_directory = os.path.join(self.target_dir, "wav")
        if not os.path.exists(self.target_dir):
            _logger.info("Creating directory...%s", self.target_dir)
            os.mkdir(self.target_dir)
        if not os.path.exists(wav_directory):
            _logger.info("Creating directory...%s", wav_directory)
            os.mkdir(wav_directory)
        return wav_directory


class GramVaaniDataSets:
    def __init__(self, target_dir, wav_directory, gram_vaani_csv):
        self.target_dir = target_dir
        self.wav_directory = wav_directory
        self.csv_data = gram_vaani_csv.data
        self.raw = pd.DataFrame(columns=["wav_filename", "wav_filesize", "transcript"])
        self.valid = pd.DataFrame(
            columns=["wav_filename", "wav_filesize", "transcript"]
        )
        self.train = pd.DataFrame(
            columns=["wav_filename", "wav_filesize", "transcript"]
        )
        self.dev = pd.DataFrame(columns=["wav_filename", "wav_filesize", "transcript"])
        self.test = pd.DataFrame(columns=["wav_filename", "wav_filesize", "transcript"])

    def create(self):
        self._convert_csv_data_to_raw_data()
        self.raw.index = range(len(self.raw.index))
        self.valid = self.raw[self._is_valid_raw_rows()]
        self.valid = self.valid.sample(frac=1).reset_index(drop=True)
        train_size, dev_size, test_size = self._calculate_data_set_sizes()
        self.train = self.valid.loc[0:train_size]
        self.dev = self.valid.loc[train_size : train_size + dev_size]
        self.test = self.valid.loc[
            train_size + dev_size : train_size + dev_size + test_size
        ]

    def _convert_csv_data_to_raw_data(self):
        self.raw[["wav_filename", "wav_filesize", "transcript"]] = self.csv_data[
            ["audio_url", "transcript", "audio_length"]
        ].swifter.apply(
            func=lambda arg: self._convert_csv_data_to_raw_data_impl(*arg),
            axis=1,
            raw=True,
        )
        self.raw.reset_index()

    def _convert_csv_data_to_raw_data_impl(self, audio_url, transcript, audio_length):
        if audio_url == "audio_url":
            return pd.Series(["wav_filename", "wav_filesize", "transcript"])
        mp3_filename = os.path.basename(audio_url)
        wav_relative_filename = os.path.join(
            "wav", os.path.splitext(os.path.basename(mp3_filename))[0] + ".wav"
        )
        wav_filesize = os.path.getsize(
            os.path.join(self.target_dir, wav_relative_filename)
        )
        transcript = validate_label(transcript)
        if None == transcript:
            transcript = ""
        return pd.Series([wav_relative_filename, wav_filesize, transcript])

    def _is_valid_raw_rows(self):
        is_valid_raw_transcripts = self._is_valid_raw_transcripts()
        is_valid_raw_wav_frames = self._is_valid_raw_wav_frames()
        is_valid_raw_row = [
            (is_valid_raw_transcript & is_valid_raw_wav_frame)
            for is_valid_raw_transcript, is_valid_raw_wav_frame in zip(
                is_valid_raw_transcripts, is_valid_raw_wav_frames
            )
        ]
        series = pd.Series(is_valid_raw_row)
        return series

    def _is_valid_raw_transcripts(self):
        return pd.Series([bool(transcript) for transcript in self.raw.transcript])

    def _is_valid_raw_wav_frames(self):
        transcripts = [str(transcript) for transcript in self.raw.transcript]
        wav_filepaths = [
            os.path.join(self.target_dir, str(wav_filename))
            for wav_filename in self.raw.wav_filename
        ]
        wav_frames = [
            int(
                subprocess.check_output(
                    ["soxi", "-s", wav_filepath], stderr=subprocess.STDOUT
                )
            )
            for wav_filepath in wav_filepaths
        ]
        is_valid_raw_wav_frames = [
            self._is_wav_frame_valid(wav_frame, transcript)
            for wav_frame, transcript in zip(wav_frames, transcripts)
        ]
        return pd.Series(is_valid_raw_wav_frames)

    def _is_wav_frame_valid(self, wav_frame, transcript):
        is_wav_frame_valid = True
        if int(wav_frame / SAMPLE_RATE * 1000 / 10 / 2) < len(str(transcript)):
            is_wav_frame_valid = False
        elif wav_frame / SAMPLE_RATE > MAX_SECS:
            is_wav_frame_valid = False
        return is_wav_frame_valid

    def _calculate_data_set_sizes(self):
        total_size = len(self.valid)
        dev_size = math.floor(total_size * DEV_PERCENTAGE)
        train_size = math.floor(total_size * TRAIN_PERCENTAGE)
        test_size = total_size - (train_size + dev_size)
        return (train_size, dev_size, test_size)

    def save(self):
        datasets = ["train", "dev", "test"]
        for dataset in datasets:
            self._save(dataset)

    def _save(self, dataset):
        dataset_path = os.path.join(self.target_dir, dataset + ".csv")
        dataframe = getattr(self, dataset)
        dataframe.to_csv(
            dataset_path,
            index=False,
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_MINIMAL,
        )


def main(args):
    """Main entry point allowing external calls
    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    validate_label = get_validate_label(args)
    setup_logging(args.loglevel)
    _logger.info("Starting GramVaani importer...")
    _logger.info("Starting loading GramVaani csv...")
    csv = GramVaaniCSV(args.csv_filename)
    _logger.info("Starting downloading GramVaani mp3's...")
    downloader = GramVaaniDownloader(csv, args.target_dir)
    mp3_directory = downloader.download()
    _logger.info("Starting converting GramVaani mp3's to wav's...")
    converter = GramVaaniConverter(args.target_dir, mp3_directory)
    wav_directory = converter.convert()
    datasets = GramVaaniDataSets(args.target_dir, wav_directory, csv)
    datasets.create()
    datasets.save()
    _logger.info("Finished GramVaani importer...")


main(sys.argv[1:])
