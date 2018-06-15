#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import sys

# To use util.tc
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(sys.argv[0]))))
import util.taskcluster as tcu
from util.benchmark import keep_only_digits

import argparse
import numpy
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
import scipy.io.wavfile as wav
import csv
import getpass

from six import iteritems
from six.moves import range, map

r'''
 Tool to:
  - ingest CSV file produced by benchmark_nc and produce nice plots
'''

def reduce_filename(f):
    r'''
    Expects something like /tmp/tmpAjry4Gdsbench/test.weights.e5.XXX.YYY.pb
    Where XXX is a variation on the model size for example
    And where YYY is a const related to the training dataset
    '''

    f = os.path.basename(f).split('.')
    if f[1] == 'aot':
        return 'AOT:' + str(keep_only_digits(f[-3]))
    else:
        return keep_only_digits(f[-3])

def ingest_csv(datasets=None, range=None):
    existing_files = filter(lambda x: os.path.isfile(x[1]), datasets)
    assert len(datasets) == len(existing_files)

    if range:
        range = map(int, range.split(','))

    data = {}
    for (dsname, dsfile) in datasets:
        print('Reading %s from %s' % (dsname, dsfile))
        with open(dsfile) as f:
            d = csv.DictReader(f)
            data[dsname] = []
            for e in d:
                if range:
                    re       = reduce_filename(e['model'])
                    in_range = (re >= range[0] and re <= range[1])
                    if in_range:
                        data[dsname].append(e)
                else:
                    data[dsname].append(e)

    return data

def produce_plot(input=None, output=None):
    x = range(len(input))
    xlabels = list(map(lambda a: a['name'], input))
    y = list(map(lambda a: a['mean'], input))
    yerr = list(map(lambda a: a['stddev'], input))

    print('y=', y)
    print('yerr=', yerr)
    plt.errorbar(x, y, yerr=yerr)
    plt.show()
    print("Wrote as %s" % output.name)

def produce_plot_multiseries(input=None, output=None, title=None, size=None, fig_dpi=None, source_wav=None):
    fig, ax = plt.subplots()
    # float() required because size.split()[] is a string
    fig.set_figwidth(float(size.split('x')[0]) / fig_dpi)
    fig.set_figheight(float(size.split('x')[1]) / fig_dpi)

    nb_items = len(input[input.keys()[0]])
    x_all    = list(range(nb_items))
    for serie, serie_values in iteritems(input):
        xtics  = list(map(lambda a: reduce_filename(a['model']), serie_values))
        y      = list(map(lambda a: float(a['mean']), serie_values))
        yerr   = list(map(lambda a: float(a['std']), serie_values))
        linreg = scipy_stats.linregress(x_all, y)
        ylin   = linreg.intercept + linreg.slope * numpy.asarray(x_all)

        ax.errorbar(x_all, y, yerr=yerr, label=('%s' % serie), fmt='-', capsize=4, elinewidth=1)
        ax.plot(x_all, ylin, label=('%s ~= %0.4f*x+%0.4f (R=%0.4f)' % (serie, linreg.slope, linreg.intercept, linreg.rvalue)))

        plt.xticks(x_all, xtics, rotation=60)

    if source_wav:
        audio = wav.read(source_wav)
        print('Adding realtime')
        for rt_factor in [ 0.5, 1.0, 1.5, 2.0 ]:
            rt_secs = len(audio[1]) / audio[0] * rt_factor
            y_rt    = numpy.repeat(rt_secs, nb_items)
            ax.plot(x_all, y_rt, label=('Realtime: %0.4f secs [%0.1f]' % (rt_secs, rt_factor)))

    ax.set_title(title)
    ax.set_xlabel('Model size')
    ax.set_ylabel('Execution time (s)')
    legend = ax.legend(loc='best')

    plot_format = os.path.splitext(output.name)[-1].split('.')[-1]

    plt.grid()
    plt.tight_layout()
    plt.savefig(output, transparent=False, frameon=True, dpi=fig_dpi, format=plot_format)

def handle_args():
    parser = argparse.ArgumentParser(description='Benchmarking tooling for DeepSpeech native_client.')
    parser.add_argument('--wav', required=False,
                                 help='WAV file to pass to native_client. Supply again in plotting mode to draw realine line.')
    parser.add_argument('--dataset', action='append', nargs=2, metavar=('name','source'),
                                help='Include dataset NAME from file SOURCE. Repeat the option to add more datasets.')
    parser.add_argument('--title', default=None, help='Title of the plot.')
    parser.add_argument('--plot', type=argparse.FileType('w'), required=False,
                                help='Target file where to plot data. Format will be deduced from extension.')
    parser.add_argument('--size', default='800x600',
                                help='Size (px) of the resulting plot.')
    parser.add_argument('--dpi', type=int, default=96,
                                help='Set plot DPI.')
    parser.add_argument('--range', default=None,
                                help='Range of model size to use. Comma-separated string of boundaries: min,max')
    return parser.parse_args()

def do_main():
    cli_args = handle_args()

    if not cli_args.dataset or not cli_args.plot:
        raise AssertionError('Missing arguments (dataset or target file)')

    # This is required to avoid errors about missing DISPLAY env var
    plt.switch_backend('agg')
    all_inference_times = ingest_csv(datasets=cli_args.dataset, range=cli_args.range)

    if cli_args.plot:
        produce_plot_multiseries(input=all_inference_times, output=cli_args.plot, title=cli_args.title, size=cli_args.size, fig_dpi=cli_args.dpi, source_wav=cli_args.wav)

if __name__ == '__main__' :
    do_main()
