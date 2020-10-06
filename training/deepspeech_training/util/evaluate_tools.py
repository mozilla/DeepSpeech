#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import json
from multiprocessing.dummy import Pool

import numpy as np
from attrdict import AttrDict

from .flags import FLAGS
from .text import levenshtein


def pmap(fun, iterable):
    pool = Pool()
    results = pool.map(fun, iterable)
    pool.close()
    return results


def wer_cer_batch(samples):
    r"""
    The WER is defined as the edit/Levenshtein distance on word level divided by
    the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    wer = sum(s.word_distance for s in samples) / sum(s.word_length for s in samples)
    cer = sum(s.char_distance for s in samples) / sum(s.char_length for s in samples)

    wer = min(wer, 1.0)
    cer = min(cer, 1.0)

    return wer, cer


def process_decode_result(item):
    wav_filename, ground_truth, prediction, loss = item
    char_distance = levenshtein(ground_truth, prediction)
    char_length = len(ground_truth)
    word_distance = levenshtein(ground_truth.split(), prediction.split())
    word_length = len(ground_truth.split())
    return AttrDict({
        'wav_filename': wav_filename,
        'src': ground_truth,
        'res': prediction,
        'loss': loss,
        'char_distance': char_distance,
        'char_length': char_length,
        'word_distance': word_distance,
        'word_length': word_length,
        'cer': char_distance / char_length,
        'wer': word_distance / word_length,
    })


def calculate_and_print_report(wav_filenames, labels, decodings, losses, dataset_name):
    r'''
    This routine will calculate and print a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = pmap(process_decode_result, zip(wav_filenames, labels, decodings, losses))

    # Getting the WER and CER from the accumulated edit distances and lengths
    samples_wer, samples_cer = wer_cer_batch(samples)

    # Reversed because the worst WER with the best loss is to identify systemic issues, where the acoustic model is confident,
    # yet the result is completely off the mark. This can point to transcription errors and stuff like that.
    samples.sort(key=lambda s: s.loss, reverse=True)

    # Then order by ascending WER/CER
    if FLAGS.bytes_output_mode:
        samples.sort(key=lambda s: s.cer)
    else:
        samples.sort(key=lambda s: s.wer)

    # Print the report
    print_report(samples, losses, samples_wer, samples_cer, dataset_name)

    return samples


def print_report(samples, losses, wer, cer, dataset_name):
    """ Print a report summary and samples of best, median and worst results """

    # Print summary
    mean_loss = np.mean(losses)
    print('Test on %s - WER: %f, CER: %f, loss: %f' % (dataset_name, wer, cer, mean_loss))
    print('-' * 80)

    best_samples = samples[:FLAGS.report_count]
    worst_samples = samples[-FLAGS.report_count:]
    median_index = int(len(samples) / 2)
    median_left = int(FLAGS.report_count / 2)
    median_right = FLAGS.report_count - median_left
    median_samples = samples[median_index - median_left:median_index + median_right]

    def print_single_sample(sample):
        print('WER: %f, CER: %f, loss: %f' % (sample.wer, sample.cer, sample.loss))
        print(' - wav: file://%s' % sample.wav_filename)
        print(' - src: "%s"' % sample.src)
        print(' - res: "%s"' % sample.res)
        print('-' * 80)

    print('Best WER:', '\n' + '-' * 80)
    for s in best_samples:
        print_single_sample(s)

    print('Median WER:', '\n' + '-' * 80)
    for s in median_samples:
        print_single_sample(s)

    print('Worst WER:', '\n' + '-' * 80)
    for s in worst_samples:
        print_single_sample(s)


def save_samples_json(samples, output_path):
    ''' Save decoded tuples as JSON, converting NumPy floats to Python floats.

        We set ensure_ascii=True to prevent json from escaping non-ASCII chars
        in the texts.
    '''
    with open(output_path, 'w') as fout:
        json.dump(samples, fout, default=float, ensure_ascii=False, indent=2)
