#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from multiprocessing.dummy import Pool

from attrdict import AttrDict

from util.text import wer_cer_batch, levenshtein

def pmap(fun, iterable):
    pool = Pool()
    results = pool.map(fun, iterable)
    pool.close()
    return results

def process_decode_result(item):
    label, decoding, loss = item
    char_distance = levenshtein(label, decoding)
    word_distance = levenshtein(label.split(), decoding.split())
    word_length = float(len(label.split()))
    return AttrDict({
        'src': label,
        'res': decoding,
        'loss': loss,
        'cer': char_distance,
        'wer': word_distance / word_length,
    })


def calculate_report(labels, decodings, losses):
    r'''
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = pmap(process_decode_result, zip(labels, decodings, losses))

    # Getting the WER and CER from the accumulated edit distances and lengths
    samples_wer, samples_cer = wer_cer_batch(labels, decodings)

    # Order the remaining items by their loss (lowest loss on top)
    samples.sort(key=lambda s: s.loss)

    # Then order by WER (highest WER on top)
    samples.sort(key=lambda s: s.wer, reverse=True)

    return samples_wer, samples_cer, samples
