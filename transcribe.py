#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.compat.v1.logging as tflogging
tflogging.set_verbosity(tflogging.ERROR)
import logging
logging.getLogger('sox').setLevel(logging.ERROR)
import glob

from deepspeech_training.util.audio import AudioFile
from deepspeech_training.util.config import Config, initialize_globals
from deepspeech_training.util.feeding import split_audio_file
from deepspeech_training.util.flags import create_flags, FLAGS
from deepspeech_training.util.logging import log_error, log_info, log_progress, create_progressbar
from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from multiprocessing import Process, cpu_count


def fail(message, code=1):
    log_error(message)
    sys.exit(code)


def transcribe_file(audio_path, tlog_path):
    from deepspeech_training.train import create_model  # pylint: disable=cyclic-import,import-outside-toplevel
    from deepspeech_training.util.checkpoints import load_graph_for_evaluation
    initialize_globals()
    scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.scorer_path, Config.alphabet)
    try:
        num_processes = cpu_count()
    except NotImplementedError:
        num_processes = 1
    with AudioFile(audio_path, as_path=True) as wav_path:
        data_set = split_audio_file(wav_path,
                                    batch_size=FLAGS.batch_size,
                                    aggressiveness=FLAGS.vad_aggressiveness,
                                    outlier_duration_ms=FLAGS.outlier_duration_ms,
                                    outlier_batch_size=FLAGS.outlier_batch_size)
        iterator = tf.data.Iterator.from_structure(data_set.output_types, data_set.output_shapes,
                                                   output_classes=data_set.output_classes)
        batch_time_start, batch_time_end, batch_x, batch_x_len = iterator.get_next()
        no_dropout = [None] * 6
        logits, _ = create_model(batch_x=batch_x, seq_length=batch_x_len, dropout=no_dropout)
        transposed = tf.nn.softmax(tf.transpose(logits, [1, 0, 2]))
        tf.train.get_or_create_global_step()
        with tf.Session(config=Config.session_config) as session:
            load_graph_for_evaluation(session)
            session.run(iterator.make_initializer(data_set))
            transcripts = []
            while True:
                try:
                    starts, ends, batch_logits, batch_lengths = \
                        session.run([batch_time_start, batch_time_end, transposed, batch_x_len])
                except tf.errors.OutOfRangeError:
                    break
                decoded = ctc_beam_search_decoder_batch(batch_logits, batch_lengths, Config.alphabet, FLAGS.beam_width,
                                                        num_processes=num_processes,
                                                        scorer=scorer)
                decoded = list(d[0][1] for d in decoded)
                transcripts.extend(zip(starts, ends, decoded))
            transcripts.sort(key=lambda t: t[0])
            transcripts = [{'start': int(start),
                            'end': int(end),
                            'transcript': transcript} for start, end, transcript in transcripts]
            with open(tlog_path, 'w') as tlog_file:
                json.dump(transcripts, tlog_file, default=float)


def transcribe_many(src_paths,dst_paths):
    pbar = create_progressbar(prefix='Transcribing files | ', max_value=len(src_paths)).start()
    for i in range(len(src_paths)):
        p = Process(target=transcribe_file, args=(src_paths[i], dst_paths[i]))
        p.start()
        p.join()
        log_progress('Transcribed file {} of {} from "{}" to "{}"'.format(i + 1, len(src_paths), src_paths[i], dst_paths[i]))
        pbar.update(i)
    pbar.finish()


def transcribe_one(src_path, dst_path):
    transcribe_file(src_path, dst_path)
    log_info('Transcribed file "{}" to "{}"'.format(src_path, dst_path))


def resolve(base_path, spec_path):
    if spec_path is None:
        return None
    if not os.path.isabs(spec_path):
        spec_path = os.path.join(base_path, spec_path)
    return spec_path


def main(_):
    if not FLAGS.src or not os.path.exists(FLAGS.src):
        # path not given or non-existant
        fail('You have to specify which file or catalog to transcribe via the --src flag.')
    else:
        # path given and exists
        src_path = os.path.abspath(FLAGS.src)
        if os.path.isfile(src_path):
            if src_path.endswith('.catalog'):
                # Transcribe batch of files via ".catalog" file (from DSAlign)
                if FLAGS.dst:
                    fail('Parameter --dst not supported if --src points to a catalog')
                catalog_dir = os.path.dirname(src_path)
                with open(src_path, 'r') as catalog_file:
                    catalog_entries = json.load(catalog_file)
                catalog_entries = [(resolve(catalog_dir, e['audio']), resolve(catalog_dir, e['tlog'])) for e in catalog_entries]
                if any(map(lambda e: not os.path.isfile(e[0]), catalog_entries)):
                    fail('Missing source file(s) in catalog')
                if not FLAGS.force and any(map(lambda e: os.path.isfile(e[1]), catalog_entries)):
                    fail('Destination file(s) from catalog already existing, use --force for overwriting')
                if any(map(lambda e: not os.path.isdir(os.path.dirname(e[1])), catalog_entries)):
                    fail('Missing destination directory for at least one catalog entry')
                src_paths,dst_paths = zip(*paths)
                transcribe_many(src_paths,dst_paths)
            else:
                # Transcribe one file
                dst_path = os.path.abspath(FLAGS.dst) if FLAGS.dst else os.path.splitext(src_path)[0] + '.tlog'
                if os.path.isfile(dst_path):
                    if FLAGS.force:
                        transcribe_one(src_path, dst_path)
                    else:
                        fail('Destination file "{}" already existing - use --force for overwriting'.format(dst_path), code=0)
                elif os.path.isdir(os.path.dirname(dst_path)):
                    transcribe_one(src_path, dst_path)
                else:
                    fail('Missing destination directory')
        elif os.path.isdir(src_path):
            # Transcribe all files in dir
            print("Transcribing all WAV files in --src")
            if FLAGS.dst:
                fail('Destination file not supported for batch decoding jobs.')
            else:
                if not FLAGS.recursive:
                    print("If you wish to recursively scan --src, then you must use --recursive")
                    wav_paths = glob.glob(src_path + "/*.wav")
                else:
                    wav_paths = glob.glob(src_path + "/**/*.wav")
                dst_paths = [path.replace('.wav','.tlog') for path in wav_paths]
                transcribe_many(wav_paths,dst_paths)


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_string('src', '', 'Source path to an audio file or directory or catalog file.'
                                          'Catalog files should be formatted from DSAlign. A directory will'
                                          'be recursively searched for audio. If --dst not set, transcription logs (.tlog) will be '
                                          'written in-place using the source filenames with '
                                          'suffix ".tlog" instead of ".wav".')
    tf.app.flags.DEFINE_string('dst', '', 'path for writing the transcription log or logs (.tlog). '
                                          'If --src is a directory, this one also has to be a directory '
                                          'and the required sub-dir tree of --src will get replicated.')
    tf.app.flags.DEFINE_boolean('recursive', False, 'scan dir of audio recursively')
    tf.app.flags.DEFINE_boolean('force', False, 'Forces re-transcribing and overwriting of already existing '
                                                'transcription logs (.tlog)')
    tf.app.flags.DEFINE_integer('vad_aggressiveness', 3, 'How aggressive (0=lowest, 3=highest) the VAD should '
                                                         'split audio')
    tf.app.flags.DEFINE_integer('batch_size', 40, 'Default batch size')
    tf.app.flags.DEFINE_float('outlier_duration_ms', 10000, 'Duration in ms after which samples are considered outliers')
    tf.app.flags.DEFINE_integer('outlier_batch_size', 1, 'Batch size for duration outliers (defaults to 1)')
    tf.app.run(main)
