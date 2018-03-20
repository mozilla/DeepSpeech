#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import itertools
import json
import numpy as np
import os
import pandas
import progressbar
import sys
import tables
import tensorflow as tf

from attrdict import AttrDict
from collections import namedtuple
from DeepSpeech import initialize_globals, create_flags, log_debug, log_info, log_warn, log_error, create_inference_graph, decode_with_lm
from multiprocessing import Pool
from six.moves import zip, range
from util.audio import audiofile_to_input_vector
from util.text import sparse_tensor_value_to_texts, text_to_char_array, Alphabet, ctc_label_dense_to_sparse, wer, levenshtein


FLAGS = tf.app.flags.FLAGS

N_STEPS = 16
N_FEATURES = 26
N_CONTEXT = 9


def pmap(fun, iterable, threads=8):
    pool = Pool(threads)
    results = pool.map(fun, iterable)
    pool.close()
    return results


def process_single_file(row):
    # row = index, Series
    _, file = row
    features = audiofile_to_input_vector(file.wav_filename, N_FEATURES, N_CONTEXT)
    transcript = text_to_char_array(file.transcript, alphabet)

    return features, len(features), transcript, len(transcript)


# load samples from CSV, compute features, optionally cache results on disk
def preprocess(dataset_files, batch_size, hdf5_dest_path=None):
    COLUMNS = ('features', 'features_len', 'transcript', 'transcript_len')

    if hdf5_dest_path and os.path.exists(hdf5_dest_path):
        with tables.open_file(hdf5_dest_path, 'r') as file:
            features = file.root.features[:]
            features_len = file.root.features_len[:]
            transcript = file.root.transcript[:]
            transcript_len = file.root.transcript_len[:]

            # features are stored flattened, so reshape into
            # [n_steps, (n_input + 2*n_context*n_input)]
            for i in range(len(features)):
                features[i] = np.reshape(features[i], [features_len[i], -1])

            in_data = list(zip(features, features_len,
                               transcript, transcript_len))
            return pandas.DataFrame(data=in_data, columns=COLUMNS)

    csv_files = dataset_files.split(',')
    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file)

    # discard last samples if dataset does not divide batch size evenly
    if len(source_data) % batch_size != 0:
        source_data = source_data[:-(len(source_data) % batch_size)]

    out_data = pmap(process_single_file, source_data.iterrows())

    if hdf5_dest_path:
        # list of tuples -> tuple of lists
        features, features_len, transcript, transcript_len = zip(*out_data)

        with tables.open_file(hdf5_dest_path, 'w') as file:
            features_dset = file.create_vlarray(file.root, 'features',
                                                tables.Float32Atom(shape=()), filters=tables.Filters(complevel=1))
            # VLArray atoms need to be 1D, so flatten feature array
            for f in features:
                features_dset.append(np.reshape(f, -1))

            features_len_dset = file.create_array(
                file.root, 'features_len', features_len)

            transcript_dset = file.create_vlarray(
                file.root,
                'transcript',
                tables.Int32Atom(),
                filters=tables.Filters(
                    complevel=1))
            for t in transcript:
                transcript_dset.append(t)

            transcript_len_dset = file.create_array(
                file.root, 'transcript_len', transcript_len)

    return pandas.DataFrame(data=out_data, columns=COLUMNS)


def split_data(dataset, batch_size):
    remainder = len(dataset) % batch_size
    if remainder != 0:
        dataset = dataset[:-remainder]

    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]


def pad_to_dense(jagged):
    maxlen = max(len(r) for r in jagged)
    subshape = jagged[0].shape

    padded = np.zeros((len(jagged), maxlen) +
                      subshape[1:], dtype=jagged[0].dtype)
    for i, row in enumerate(jagged):
        padded[i, :len(row)] = row
    return padded


def process_decode_result(item):
    label, decoding, distance, loss = item
    sample_wer = wer(label, decoding)
    return AttrDict({
        'src': label,
        'res': decoding,
        'loss': loss,
        'distance': distance,
        'wer': sample_wer,
        'levenshtein': levenshtein(label.split(), decoding.split()),
        'label_length': float(len(label.split())),
    })


def calculate_report(labels, decodings, distances, losses):
    r'''
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = pmap(process_decode_result, zip(labels, decodings, distances, losses))

    total_levenshtein = sum(s.levenshtein for s in samples)
    total_label_length = sum(s.label_length for s in samples)

    # Getting the WER from the accumulated levenshteins and lengths
    samples_wer = total_levenshtein / total_label_length

    # Order the remaining items by their loss (lowest loss on top)
    samples.sort(key=lambda s: s.loss)

    # Then order by WER (lowest WER on top)
    samples.sort(key=lambda s: s.wer)

    return samples_wer, samples


def main(_):
    initialize_globals()

    if not FLAGS.test_files:
        log_error('You need to specify what files to use for evaluation via '
                  'the --test_files flag.')
        exit(1)

    global alphabet
    alphabet = Alphabet(os.path.abspath(FLAGS.alphabet_config_path))

    # sort examples by length, improves packing of batches and timesteps
    test_data = preprocess(
        FLAGS.test_files,
        FLAGS.test_batch_size,
        hdf5_dest_path=FLAGS.hdf5_test_set).sort_values(
        by="features_len",
        ascending=False)

    with tf.Session() as session:
        inputs, outputs = create_inference_graph(batch_size=FLAGS.test_batch_size, n_steps=N_STEPS)

        seq_lengths_ph = tf.placeholder(tf.int32, [FLAGS.test_batch_size])
        decode_logits_ph = tf.placeholder(tf.float32, [None, FLAGS.test_batch_size, alphabet.size() + 1])
        labels_ph = tf.placeholder(tf.int32, [FLAGS.test_batch_size, None])
        label_lengths_ph = tf.placeholder(tf.int32, [FLAGS.test_batch_size])

        decoded, _ = decode_with_lm(decode_logits_ph,
                                    seq_lengths_ph,
                                    merge_repeated=False,
                                    beam_width=FLAGS.beam_width)

        sparse_labels = tf.cast(
            ctc_label_dense_to_sparse(labels_ph, label_lengths_ph, FLAGS.test_batch_size),
            tf.int32)
        loss = tf.nn.ctc_loss(labels=sparse_labels,
                              inputs=decode_logits_ph,
                              sequence_length=seq_lengths_ph)

        distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), sparse_labels)

        # Create a saver using variables from the above newly created graph
        mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        saver = tf.train.Saver(mapping)

        # Restore variables from training checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error(
                'Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(
                    FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        logitses = []

        batch_count = len(test_data) // FLAGS.test_batch_size
        bar = progressbar.ProgressBar(
            max_value=batch_count - 1,
            widget=progressbar.AdaptiveETA)

        for batch in bar(split_data(test_data, FLAGS.test_batch_size)):
            session.run(outputs['initialize_state'])

            batch_features = pad_to_dense(batch['features'].values)
            batch_features_len = batch['features_len'].values
            full_step_len = np.full_like(batch_features_len, N_STEPS)

            logits = np.empty([0, FLAGS.test_batch_size, alphabet.size() + 1])
            for i in range(0, batch_features.shape[1], N_STEPS):
                chunk_features = batch_features[:, i:i + N_STEPS, :]
                chunk_features_len = np.minimum(
                    batch_features_len, full_step_len)

                # pad with zeros if the chunk does not have enough steps
                steps_in_chunk = chunk_features.shape[1]
                if steps_in_chunk < FLAGS.n_steps:
                    chunk_features = np.pad(chunk_features,
                                            ((0, 0),
                                             (0, FLAGS.n_steps - steps_in_chunk),
                                             (0, 0)),
                                            mode='constant',
                                            constant_values=0)

                output = session.run(outputs['outputs'], feed_dict={
                    inputs['input']: chunk_features,
                    inputs['input_lengths']: chunk_features_len,
                })
                logits = np.concatenate((logits, output))

                # we have processed N_STEPS so subtract from remaining steps
                batch_features_len -= N_STEPS
                # clip to zero
                batch_features_len = np.maximum(batch_features_len, np.zeros_like(batch_features_len))

            logitses.append(logits)

        ground_truths = []
        predictions = []
        distances = []
        losses = []

        bar = progressbar.ProgressBar(max_value=batch_count - 1,
                                      widget=progressbar.AdaptiveETA)

        for logits, batch in bar(zip(logitses, split_data(test_data, FLAGS.test_batch_size))):
            seq_lengths = batch['features_len'].values
            labels = pad_to_dense(batch['transcript'].values)
            label_lengths = batch['transcript_len'].values

            decoded_, loss_, distance_, sparse_labels_ = session.run([decoded, loss, distance, sparse_labels], feed_dict={
                decode_logits_ph: logits,
                seq_lengths_ph: seq_lengths,
                labels_ph: labels,
                label_lengths_ph: label_lengths
            })

            ground_truths.extend(sparse_tensor_value_to_texts(sparse_labels_, alphabet))
            predictions.extend(sparse_tensor_value_to_texts(decoded_[0], alphabet))
            distances.extend(distance_)
            losses.extend(loss_)

    wer, samples = calculate_report(ground_truths, predictions, distances, losses)
    mean_edit_distance = np.mean(distances)
    mean_loss = np.mean(losses)

    # Filter out all items with WER=0 and take only the first report_count items
    report_samples = itertools.islice((s for s in samples if s.wer > 0), FLAGS.report_count)

    print('Test - WER: %f, loss: %f, mean edit distance: %f' %
          (wer, mean_loss, mean_edit_distance))
    print('-' * 80)
    for sample in report_samples:
        print('WER: %f, loss: %f, mean edit distance: %f' %
              (sample.wer, sample.loss, sample.distance))
        print(' - src: "%s"' % sample.src)
        print(' - res: "%s"' % sample.res)
        print('-' * 80)

    if FLAGS.test_output_file:
        json.dump(samples, open(FLAGS.test_output_file, 'w'), default=lambda x: float(x))


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_string('hdf5_test_set', '', 'path to hdf5 file to cache test set features')
    tf.app.flags.DEFINE_string('test_output_file', '', 'path to a file to save all src/decoded/distance/loss tuples')
    tf.app.run(main)
