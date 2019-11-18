#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import gc
import sys
import json
import tensorflow as tf

from multiprocessing import cpu_count
from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from util.config import Config, initialize_globals
from util.audio import AudioFile
from util.feeding import split_audio_file
from util.flags import create_flags, FLAGS
from util.logging import log_error, log_info, log_progress, create_progressbar


def split_audio_file_flags(audio_file):
    return split_audio_file(audio_file,
                            batch_size=FLAGS.batch_size,
                            aggressiveness=FLAGS.vad_aggressiveness,
                            outlier_duration_ms=FLAGS.outlier_duration_ms,
                            outlier_batch_size=FLAGS.outlier_batch_size)


def transcribe(path_pairs, create_model, try_loading):
    scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.lm_binary_path, FLAGS.lm_trie_path, Config.alphabet)
    audio_path, _ = path_pairs[0]
    data_set = split_audio_file_flags(None)
    iterator = tf.data.Iterator.from_structure(data_set.output_types, data_set.output_shapes,
                                               output_classes=data_set.output_classes)
    batch_time_start, batch_time_end, batch_x, batch_x_len = iterator.get_next()
    no_dropout = [None] * 6
    logits, _ = create_model(batch_x=batch_x, seq_length=batch_x_len, dropout=no_dropout)
    transposed = tf.nn.softmax(tf.transpose(logits, [1, 0, 2]))
    tf.train.get_or_create_global_step()
    try:
        num_processes = cpu_count()
    except NotImplementedError:
        num_processes = 1
    saver = tf.train.Saver()

    with tf.Session(config=Config.session_config) as session:
        loaded = try_loading(session, saver, 'best_dev_checkpoint', 'best validation')
        if not loaded:
            loaded = try_loading(session, saver, 'checkpoint', 'most recent')
        if not loaded:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'
                      .format(FLAGS.checkpoint_dir))
            sys.exit(1)

        def run_transcription(p_index, p_data_set, p_audio_path, p_tlog_path):
            bar = create_progressbar(prefix='Transcribing file {} "{}" | '.format(p_index, p_audio_path)).start()
            log_progress('Transcribing file {}, "{}"...'.format(p_index, p_audio_path))
            session.run(iterator.make_initializer(p_data_set))
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
                bar.update(len(transcripts))
            bar.finish()
            transcripts.sort(key=lambda t: t[0])
            transcripts = [{'start': int(start),
                            'end': int(end),
                            'transcript': transcript} for start, end, transcript in transcripts]
            log_info('Writing transcript log to "{}"...'.format(p_tlog_path))
            with open(p_tlog_path, 'w') as tlog_file:
                json.dump(transcripts, tlog_file, default=float)

        for index, (audio_path, tlog_path) in enumerate(path_pairs):
            with AudioFile(audio_path, as_path=True) as wav_path:
                data_set = split_audio_file_flags(wav_path)
                run_transcription(index, data_set, audio_path, tlog_path)
                gc.collect()


def resolve(base_path, spec_path):
    if spec_path is None:
        return None
    if not os.path.isabs(spec_path):
        spec_path = os.path.join(base_path, spec_path)
    return spec_path


def main(_):
    initialize_globals()

    if not FLAGS.src:
        log_error('You have to specify which file or catalog to transcribe via the --src flag.')
        sys.exit(1)

    from DeepSpeech import create_model, try_loading  # pylint: disable=cyclic-import,import-outside-toplevel

    src_path = os.path.abspath(FLAGS.src)
    if not os.path.isfile(src_path):
        log_error('Path in --src not existing')
        sys.exit(1)
    if src_path.endswith('.catalog'):
        if FLAGS.dst:
            log_error('Parameter --dst not supported if --src points to a catalog')
            sys.exit(1)
        catalog_dir = os.path.dirname(src_path)
        with open(src_path, 'r') as catalog_file:
            catalog_entries = json.load(catalog_file)
        catalog_entries = [(resolve(catalog_dir, e['audio']), resolve(catalog_dir, e['tlog'])) for e in catalog_entries]
        if any(map(lambda e: not os.path.isfile(e[0]), catalog_entries)):
            log_error('Missing source file(s) in catalog')
            sys.exit(1)
        if not FLAGS.force and any(map(lambda e: os.path.isfile(e[1]), catalog_entries)):
            log_error('Destination file(s) from catalog already existing, use --force for overwriting')
            sys.exit(1)
        if any(map(lambda e: not os.path.isdir(os.path.dirname(e[1])), catalog_entries)):
            log_error('Missing destination directory for at least one catalog entry')
            sys.exit(1)
        transcribe(catalog_entries, create_model, try_loading)
    else:
        dst_path = os.path.abspath(FLAGS.dst) if FLAGS.dst else os.path.splitext(src_path)[0] + '.tlog'
        if os.path.isfile(dst_path):
            if FLAGS.force:
                transcribe([(src_path, dst_path)], create_model, try_loading)
            else:
                log_error('Destination file "{}" already existing - requires --force for overwriting'.format(dst_path))
                sys.exit(0)
        elif os.path.isdir(os.path.dirname(dst_path)):
            transcribe([(src_path, dst_path)], create_model, try_loading)
        else:
            log_error('Missing destination directory')
            sys.exit(1)


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_string('src', '', 'source path to an audio file or directory to recursively scan '
                                          'for audio files. If --dst not set, transcription logs (.tlog) will be '
                                          'written in-place using the source filenames with '
                                          'suffix ".tlog" instead of ".wav".')
    tf.app.flags.DEFINE_string('dst', '', 'path for writing the transcription log or logs (.tlog). '
                                          'If --src is a directory, this one also has to be a directory '
                                          'and the required sub-dir tree of --src will get replicated.')
    tf.app.flags.DEFINE_boolean('force', False, 'Forces re-transcribing and overwriting of already existing '
                                                'transcription logs (.tlog)')
    tf.app.flags.DEFINE_integer('vad_aggressiveness', 3, 'How aggressive (0=lowest, 3=highest) the VAD should '
                                                         'split audio')
    tf.app.flags.DEFINE_integer('batch_size', 40, 'Default batch size')
    tf.app.flags.DEFINE_float('outlier_duration_ms', 10000, 'Duration in ms after which samples are considered outliers')
    tf.app.flags.DEFINE_integer('outlier_batch_size', 1, 'Batch size for duration outliers (defaults to 1)')
    tf.app.run(main)
