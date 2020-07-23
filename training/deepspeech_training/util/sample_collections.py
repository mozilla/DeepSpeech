# -*- coding: utf-8 -*-
import os
import csv
import json

from pathlib import Path
from functools import partial

from .helpers import KILOBYTE, MEGABYTE, GIGABYTE, Interleaved
from .audio import (
    Sample,
    DEFAULT_FORMAT,
    AUDIO_TYPE_PCM,
    AUDIO_TYPE_OPUS,
    SERIALIZABLE_AUDIO_TYPES,
    get_audio_type_from_extension,
    write_wav
)

BIG_ENDIAN = 'big'
INT_SIZE = 4
BIGINT_SIZE = 2 * INT_SIZE
MAGIC = b'SAMPLEDB'

BUFFER_SIZE = 1 * MEGABYTE
REVERSE_BUFFER_SIZE = 16 * KILOBYTE
CACHE_SIZE = 1 * GIGABYTE

SCHEMA_KEY = 'schema'
CONTENT_KEY = 'content'
MIME_TYPE_KEY = 'mime-type'
MIME_TYPE_TEXT = 'text/plain'
CONTENT_TYPE_SPEECH = 'speech'
CONTENT_TYPE_TRANSCRIPT = 'transcript'


class LabeledSample(Sample):
    """In-memory labeled audio sample representing an utterance.
    Derived from util.audio.Sample and used by sample collection readers and writers."""
    def __init__(self, audio_type, raw_data, transcript, audio_format=DEFAULT_FORMAT, sample_id=None):
        """
        Parameters
        ----------
        audio_type : str
            See util.audio.Sample.__init__ .
        raw_data : binary
            See util.audio.Sample.__init__ .
        transcript : str
            Transcript of the sample's utterance
        audio_format : tuple
            See util.audio.Sample.__init__ .
        sample_id : str
            Tracking ID - should indicate sample's origin as precisely as possible.
            It is typically assigned by collection readers.
        """
        super().__init__(audio_type, raw_data, audio_format=audio_format, sample_id=sample_id)
        self.transcript = transcript


def load_sample(filename, label=None):
    """
    Loads audio-file as a (labeled or unlabeled) sample

    Parameters
    ----------
    filename : str
        Filename of the audio-file to load as sample
    label : str
        Label (transcript) of the sample.
        If None: return util.audio.Sample instance
        Otherwise: return util.sample_collections.LabeledSample instance

    Returns
    -------
    util.audio.Sample instance if label is None, else util.sample_collections.LabeledSample instance
    """
    ext = os.path.splitext(filename)[1].lower()
    audio_type = get_audio_type_from_extension(ext)
    if audio_type is None:
        raise ValueError('Unknown audio type extension "{}"'.format(ext))
    with open(filename, 'rb') as audio_file:
        if label is None:
            return Sample(audio_type, audio_file.read(), sample_id=filename)
        return LabeledSample(audio_type, audio_file.read(), label, sample_id=filename)


class DirectSDBWriter:
    """Sample collection writer for creating a Sample DB (SDB) file"""
    def __init__(self,
                 sdb_filename,
                 buffering=BUFFER_SIZE,
                 audio_type=AUDIO_TYPE_OPUS,
                 bitrate=None,
                 id_prefix=None,
                 labeled=True):
        """
        Parameters
        ----------
        sdb_filename : str
            Path to the SDB file to write
        buffering : int
            Write-buffer size to use while writing the SDB file
        audio_type : str
            See util.audio.Sample.__init__ .
        bitrate : int
            Bitrate for sample-compression in case of lossy audio_type (e.g. AUDIO_TYPE_OPUS)
        id_prefix : str
            Prefix for IDs of written samples - defaults to sdb_filename
        labeled : bool or None
            If True: Writes labeled samples (util.sample_collections.LabeledSample) only.
            If False: Ignores transcripts (if available) and writes (unlabeled) util.audio.Sample instances.
        """
        self.sdb_filename = sdb_filename
        self.id_prefix = sdb_filename if id_prefix is None else id_prefix
        self.labeled = labeled
        if audio_type not in SERIALIZABLE_AUDIO_TYPES:
            raise ValueError('Audio type "{}" not supported'.format(audio_type))
        self.audio_type = audio_type
        self.bitrate = bitrate
        self.sdb_file = open(sdb_filename, 'wb', buffering=buffering)
        self.offsets = []
        self.num_samples = 0

        self.sdb_file.write(MAGIC)

        schema_entries = [{CONTENT_KEY: CONTENT_TYPE_SPEECH, MIME_TYPE_KEY: audio_type}]
        if self.labeled:
            schema_entries.append({CONTENT_KEY: CONTENT_TYPE_TRANSCRIPT, MIME_TYPE_KEY: MIME_TYPE_TEXT})
        meta_data = {SCHEMA_KEY: schema_entries}
        meta_data = json.dumps(meta_data).encode()
        self.write_big_int(len(meta_data))
        self.sdb_file.write(meta_data)

        self.offset_samples = self.sdb_file.tell()
        self.sdb_file.seek(2 * BIGINT_SIZE, 1)

    def write_int(self, n):
        return self.sdb_file.write(n.to_bytes(INT_SIZE, BIG_ENDIAN))

    def write_big_int(self, n):
        return self.sdb_file.write(n.to_bytes(BIGINT_SIZE, BIG_ENDIAN))

    def __enter__(self):
        return self

    def add(self, sample):
        def to_bytes(n):
            return n.to_bytes(INT_SIZE, BIG_ENDIAN)
        sample.change_audio_type(self.audio_type, bitrate=self.bitrate)
        opus = sample.audio.getbuffer()
        opus_len = to_bytes(len(opus))
        if self.labeled:
            transcript = sample.transcript.encode()
            transcript_len = to_bytes(len(transcript))
            entry_len = to_bytes(len(opus_len) + len(opus) + len(transcript_len) + len(transcript))
            buffer = b''.join([entry_len, opus_len, opus, transcript_len, transcript])
        else:
            entry_len = to_bytes(len(opus_len) + len(opus))
            buffer = b''.join([entry_len, opus_len, opus])
        self.offsets.append(self.sdb_file.tell())
        self.sdb_file.write(buffer)
        sample.sample_id = '{}:{}'.format(self.id_prefix, self.num_samples)
        self.num_samples += 1
        return sample.sample_id

    def close(self):
        if self.sdb_file is None:
            return
        offset_index = self.sdb_file.tell()
        self.sdb_file.seek(self.offset_samples)
        self.write_big_int(offset_index - self.offset_samples - BIGINT_SIZE)
        self.write_big_int(self.num_samples)

        self.sdb_file.seek(offset_index + BIGINT_SIZE)
        self.write_big_int(self.num_samples)
        for offset in self.offsets:
            self.write_big_int(offset)
        offset_end = self.sdb_file.tell()
        self.sdb_file.seek(offset_index)
        self.write_big_int(offset_end - offset_index - BIGINT_SIZE)
        self.sdb_file.close()
        self.sdb_file = None

    def __len__(self):
        return len(self.offsets)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SDB:  # pylint: disable=too-many-instance-attributes
    """Sample collection reader for reading a Sample DB (SDB) file"""
    def __init__(self,
                 sdb_filename,
                 buffering=BUFFER_SIZE,
                 id_prefix=None,
                 labeled=True,
                 reverse=False):
        """
        Parameters
        ----------
        sdb_filename : str
            Path to the SDB file to read samples from
        buffering : int
            Read-ahead buffer size to use while reading the SDB file in normal order. Fixed to 16kB if in reverse-mode.
        id_prefix : str
            Prefix for IDs of read samples - defaults to sdb_filename
        labeled : bool or None
            If True: Reads util.sample_collections.LabeledSample instances. Fails, if SDB file provides no transcripts.
            If False: Ignores transcripts (if available) and reads (unlabeled) util.audio.Sample instances.
            If None: Automatically determines if SDB schema has transcripts
            (reading util.sample_collections.LabeledSample instances) or not (reading util.audio.Sample instances).
        """
        self.sdb_filename = sdb_filename
        self.id_prefix = sdb_filename if id_prefix is None else id_prefix
        self.sdb_file = open(sdb_filename, 'rb', buffering=REVERSE_BUFFER_SIZE if reverse else buffering)
        self.offsets = []
        if self.sdb_file.read(len(MAGIC)) != MAGIC:
            raise RuntimeError('No Sample Database')
        meta_chunk_len = self.read_big_int()
        self.meta = json.loads(self.sdb_file.read(meta_chunk_len).decode())
        if SCHEMA_KEY not in self.meta:
            raise RuntimeError('Missing schema')
        self.schema = self.meta[SCHEMA_KEY]

        speech_columns = self.find_columns(content=CONTENT_TYPE_SPEECH, mime_type=SERIALIZABLE_AUDIO_TYPES)
        if not speech_columns:
            raise RuntimeError('No speech data (missing in schema)')
        self.speech_index = speech_columns[0]
        self.audio_type = self.schema[self.speech_index][MIME_TYPE_KEY]

        self.transcript_index = None
        if labeled is not False:
            transcript_columns = self.find_columns(content=CONTENT_TYPE_TRANSCRIPT, mime_type=MIME_TYPE_TEXT)
            if transcript_columns:
                self.transcript_index = transcript_columns[0]
            else:
                if labeled is True:
                    raise RuntimeError('No transcript data (missing in schema)')

        sample_chunk_len = self.read_big_int()
        self.sdb_file.seek(sample_chunk_len + BIGINT_SIZE, 1)
        num_samples = self.read_big_int()
        for _ in range(num_samples):
            self.offsets.append(self.read_big_int())
        if reverse:
            self.offsets.reverse()

    def read_int(self):
        return int.from_bytes(self.sdb_file.read(INT_SIZE), BIG_ENDIAN)

    def read_big_int(self):
        return int.from_bytes(self.sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)

    def find_columns(self, content=None, mime_type=None):
        criteria = []
        if content is not None:
            criteria.append((CONTENT_KEY, content))
        if mime_type is not None:
            criteria.append((MIME_TYPE_KEY, mime_type))
        if len(criteria) == 0:
            raise ValueError('At least one of "content" or "mime-type" has to be provided')
        matches = []
        for index, column in enumerate(self.schema):
            matched = 0
            for field, value in criteria:
                if column[field] == value or (isinstance(value, list) and column[field] in value):
                    matched += 1
            if matched == len(criteria):
                matches.append(index)
        return matches

    def read_row(self, row_index, *columns):
        columns = list(columns)
        column_data = [None] * len(columns)
        found = 0
        if not 0 <= row_index < len(self.offsets):
            raise ValueError('Wrong sample index: {} - has to be between 0 and {}'
                             .format(row_index, len(self.offsets) - 1))
        self.sdb_file.seek(self.offsets[row_index] + INT_SIZE)
        for index in range(len(self.schema)):
            chunk_len = self.read_int()
            if index in columns:
                column_data[columns.index(index)] = self.sdb_file.read(chunk_len)
                found += 1
                if found == len(columns):
                    return tuple(column_data)
            else:
                self.sdb_file.seek(chunk_len, 1)
        return tuple(column_data)

    def __getitem__(self, i):
        sample_id = '{}:{}'.format(self.id_prefix, i)
        if self.transcript_index is None:
            [audio_data] = self.read_row(i, self.speech_index)
            return Sample(self.audio_type, audio_data, sample_id=sample_id)
        audio_data, transcript = self.read_row(i, self.speech_index, self.transcript_index)
        transcript = transcript.decode()
        return LabeledSample(self.audio_type, audio_data, transcript, sample_id=sample_id)

    def __iter__(self):
        for i in range(len(self.offsets)):
            yield self[i]

    def __len__(self):
        return len(self.offsets)

    def close(self):
        if self.sdb_file is not None:
            self.sdb_file.close()

    def __del__(self):
        self.close()


class CSVWriter:  # pylint: disable=too-many-instance-attributes
    """Sample collection writer for writing a CSV data-set and all its referenced WAV samples"""
    def __init__(self,
                 csv_filename,
                 absolute_paths=False,
                 labeled=True):
        """
        Parameters
        ----------
        csv_filename : str
            Path to the CSV file to write.
            Will create a directory (CSV-filename without extension) next to it and fail if it already exists.
        absolute_paths : bool
            If paths in CSV file should be absolute instead of relative to the CSV file's parent directory.
        labeled : bool or None
            If True: Writes labeled samples (util.sample_collections.LabeledSample) only.
            If False: Ignores transcripts (if available) and writes (unlabeled) util.audio.Sample instances.
        """
        self.csv_filename = Path(csv_filename)
        self.csv_base_dir = self.csv_filename.parent.resolve().absolute()
        self.set_name = self.csv_filename.stem
        self.csv_dir = self.csv_base_dir / self.set_name
        if self.csv_dir.exists():
            raise RuntimeError('"{}" already existing'.format(self.csv_dir))
        os.mkdir(str(self.csv_dir))
        self.absolute_paths = absolute_paths
        fieldnames = ['wav_filename', 'wav_filesize']
        self.labeled = labeled
        if labeled:
            fieldnames.append('transcript')
        self.csv_file = open(csv_filename, 'w', encoding='utf-8', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.counter = 0

    def __enter__(self):
        return self

    def add(self, sample):
        sample_filename = self.csv_dir / 'sample{0:08d}.wav'.format(self.counter)
        self.counter += 1
        sample.change_audio_type(AUDIO_TYPE_PCM)
        write_wav(str(sample_filename), sample.audio, audio_format=sample.audio_format)
        sample.sample_id = str(sample_filename.relative_to(self.csv_base_dir))
        row = {
            'wav_filename': str(sample_filename.absolute()) if self.absolute_paths else sample.sample_id,
            'wav_filesize': sample_filename.stat().st_size
        }
        if self.labeled:
            row['transcript'] = sample.transcript
        self.csv_writer.writerow(row)
        return sample.sample_id

    def close(self):
        if self.csv_file:
            self.csv_file.close()

    def __len__(self):
        return self.counter

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SampleList:
    """Sample collection base class with samples loaded from a list of in-memory paths."""
    def __init__(self, samples, labeled=True, reverse=False):
        """
        Parameters
        ----------
        samples : iterable of tuples of the form (sample_filename, filesize [, transcript])
            File-size is used for ordering the samples; transcript has to be provided if labeled=True
        labeled : bool or None
            If True: Reads LabeledSample instances.
            If False: Ignores transcripts (if available) and reads (unlabeled) util.audio.Sample instances.
        reverse : bool
            If the order of the samples should be reversed
        """
        self.labeled = labeled
        self.samples = list(samples)
        self.samples.sort(key=lambda r: r[1], reverse=reverse)

    def __getitem__(self, i):
        sample_spec = self.samples[i]
        return load_sample(sample_spec[0], label=sample_spec[2] if self.labeled else None)

    def __len__(self):
        return len(self.samples)


class CSV(SampleList):
    """Sample collection reader for reading a DeepSpeech CSV file
    Automatically orders samples by CSV column wav_filesize (if available)."""
    def __init__(self, csv_filename, labeled=None, reverse=False):
        """
        Parameters
        ----------
        csv_filename : str
            Path to the CSV file containing sample audio paths and transcripts
        labeled : bool or None
            If True: Reads LabeledSample instances. Fails, if CSV file has no transcript column.
            If False: Ignores transcripts (if available) and reads (unlabeled) util.audio.Sample instances.
            If None: Automatically determines if CSV file has a transcript column
            (reading util.sample_collections.LabeledSample instances) or not (reading util.audio.Sample instances).
        reverse : bool
            If the order of the samples should be reversed
        """
        rows = []
        csv_dir = Path(csv_filename).parent
        with open(csv_filename, 'r', encoding='utf8') as csv_file:
            reader = csv.DictReader(csv_file)
            if 'transcript' in reader.fieldnames:
                if labeled is None:
                    labeled = True
            elif labeled:
                raise RuntimeError('No transcript data (missing CSV column)')
            for row in reader:
                wav_filename = Path(row['wav_filename'])
                if not wav_filename.is_absolute():
                    wav_filename = csv_dir / wav_filename
                wav_filename = str(wav_filename)
                wav_filesize = int(row['wav_filesize']) if 'wav_filesize' in row else 0
                if labeled:
                    rows.append((wav_filename, wav_filesize, row['transcript']))
                else:
                    rows.append((wav_filename, wav_filesize))
        super(CSV, self).__init__(rows, labeled=labeled, reverse=reverse)


def samples_from_source(sample_source, buffering=BUFFER_SIZE, labeled=None, reverse=False):
    """
    Loads samples from a sample source file.

    Parameters
    ----------
    sample_source : str
        Path to the sample source file (SDB or CSV)
    buffering : int
        Read-buffer size to use while reading files
    labeled : bool or None
        If True: Reads LabeledSample instances. Fails, if source provides no transcripts.
        If False: Ignores transcripts (if available) and reads (unlabeled) util.audio.Sample instances.
        If None: Automatically determines if source provides transcripts
        (reading util.sample_collections.LabeledSample instances) or not (reading util.audio.Sample instances).
    reverse : bool
        If the order of the samples should be reversed

    Returns
    -------
    iterable of util.sample_collections.LabeledSample or util.audio.Sample instances supporting len.
    """
    ext = os.path.splitext(sample_source)[1].lower()
    if ext == '.sdb':
        return SDB(sample_source, buffering=buffering, labeled=labeled, reverse=reverse)
    if ext == '.csv':
        return CSV(sample_source, labeled=labeled, reverse=reverse)
    raise ValueError('Unknown file type: "{}"'.format(ext))


def samples_from_sources(sample_sources, buffering=BUFFER_SIZE, labeled=None, reverse=False):
    """
    Loads and combines samples from a list of source files. Sources are combined in an interleaving way to
    keep default sample order from shortest to longest.

    Parameters
    ----------
    sample_sources : list of str
        Paths to sample source files (SDBs or CSVs)
    buffering : int
        Read-buffer size to use while reading files
    labeled : bool or None
        If True: Reads LabeledSample instances. Fails, if not all sources provide transcripts.
        If False: Ignores transcripts (if available) and always reads (unlabeled) util.audio.Sample instances.
        If None: Reads util.sample_collections.LabeledSample instances from sources with transcripts and
        util.audio.Sample instances from sources with no transcripts.
    reverse : bool
        If the order of the samples should be reversed

    Returns
    -------
    iterable of util.sample_collections.LabeledSample (labeled=True) or util.audio.Sample (labeled=False) supporting len
    """
    sample_sources = list(sample_sources)
    if len(sample_sources) == 0:
        raise ValueError('No files')
    if len(sample_sources) == 1:
        return samples_from_source(sample_sources[0], buffering=buffering, labeled=labeled, reverse=reverse)
    cols = [samples_from_source(source, buffering=buffering, labeled=labeled, reverse=reverse)
            for source in sample_sources]
    return Interleaved(*cols, key=lambda s: s.duration, reverse=reverse)
