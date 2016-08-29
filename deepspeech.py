#!/usr/bin/env python

from __future__ import print_function

## Data Import
# The import routines for the [TED-LIUM](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus) have yet to be written.

#from ted_lium import input_data
#ted_lium = input_data.read_data_sets(\"./TEDLIUM_release2\")

## Imports
# Here we first import all of the packages we require to implement the DeepSpeech BRNN.
import numpy as np

from utils import sparse_tuple_from as sparse_tuple_from

from DeepSpeech import Input, Network, Trainer, Plotter

import sys
import argparse

parser = argparse.ArgumentParser(
  description='Some basic DeepSpeech training',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
  '--runs', type=int, default=1,
  help='How many times should we run this to benchmark')

parser.add_argument(
  '--ldc93s1', type=int, default=0,
  help='How many LDC93S1 audio samples to use')

parser.add_argument(
  '--training-iters', type=int, default=250,
  help='Amount of iterations to use for training the network')

ctc = parser.add_mutually_exclusive_group(required=True)
ctc.add_argument(
  '--warpctc', dest="use_warpctc", action='store_true',
  help='Use the Warp-CTC loss function')
ctc.add_argument(
  '--ctc', dest="use_warpctc", action='store_false',
  help='Use the classic CTC loss function')

parser.add_argument(
  '--learning-rate', type=float, default=0.0001,
  help='Set the learning rate of the network for the Adam Optimizer')
parser.add_argument(
  '--beta1', type=float, default=0.9,
  help='Set beta1 parameter for the Adam Optimizer')
parser.add_argument(
  '--beta2', type=float, default=0.999,
  help='Set beta2 parameter for the Adam Optimizer')
parser.add_argument(
  '--epsilon', type=float, default=1e-8,
  help='Set epsilon parameter for the Adam Optimizer')

parser.add_argument(
  '--dropout-rate', type=float, default=0.05,
  help='Set the network dropout rate')

parser.add_argument(
  '--relu-clip', type=int, default=20,
  help='Set the ReLU clipping value for the network')

parser.add_argument(
  '--n-context', type=int, default=5,
  help='Amount of contextual data for feeding network. DeepSpeech paper suggests 5, 7 or 9')

parser.add_argument(
  '--n-character', type=int, default=28,
  help='How many characters in this alphabet')

parser.add_argument(
  '--log', type=argparse.FileType('w'), default=sys.stdout,
  help='Where to print output')

parser.add_argument(
  '--csv', type=argparse.FileType('w'),
  required=True, help='Where to save computed data')

parser.add_argument(
  '--plot', type=str, default=None,
  help='File name where to plot resulting learning of the network')

args = parser.parse_args()

print('\n+--------------------------------------+\n'
    + ' Starting instance with:\n'
    + '\ttraining_iters:\t%d\n' % args.training_iters
    + '\tuse_warpctc:\t%d\n' % args.use_warpctc
    + '\tlearning_rate:\t%.12f\n' % args.learning_rate
    + '\tbeta1:\t\t%.12f\n' % args.beta1
    + '\tbeta2:\t\t%.12f\n' % args.beta2
    + '\tepsilon:\t%.12f\n' % args.epsilon
    + '\tdropout_rate:\t%.12f\n' % args.dropout_rate
    + '\trelu_clip:\t%d\n' % args.relu_clip
    + '\tn_context:\t%d\n' % args.n_context
    + '\tn_character:\t%d\n' % args.n_character
    + '+--------------------------------------+\n')

## Global Constants
# Next we introduce several constants used in the algorithm below.  In particular, we define
# * `learning_rate` - The learning rate we will employ in Adam optimizer[[3]](http://arxiv.org/abs/1412.6980)
# * `training_iters`- The number of iterations we will train for
# * `batch_size`- The number of elements in a batch
# * `display_step`- The number of iterations we cycle through before displaying progress
##batch_size = 1        # TODO: Determine a reasonable value for this
display_step = 1       # TODO: Determine a reasonable value for this

## Geometric Constants
##>> How many times slices are in the sample
##n_steps = 291 # TODO: Determine this programatically from the longest speech sample

# Each of the `n_steps` vectors is the Fourier transform of a time-slice of the speech sample. The number of "bins" of this Fourier transform is dependent upon the sample rate of the data set.
# Generically, if the sample rate is 8kHz we use 80bins. If the sample rate is 16kHz we use 160bins... We capture the dimension of these vectors, equivalently the number of bins in the Fourier transform, in the variable `n_input`
##>> Dimension of the MFCC features vector
##n_input = 20 # TODO: Determine this programatically from the sample rate

# Next, we introduce an additional variable `n_character` which holds the number of characters in the target language plus one, for the $blank$.
# For English it is the cardinality of the set $\\{a,b,c, . . . , z, space, apostrophe, blank\\}$ we referred to earlier.

# Loading the data

def mk_ldc93s1():
  return { 'input': ['LDC93S1.wav', 93638], 'target': ['LDC93S1.txt', 62] }

audio_samples = [ mk_ldc93s1() for _as in xrange(args.ldc93s1) ]
batch_size = len(audio_samples)
print("Continuing with batch_size=%d" % batch_size)

inputs  = list()
targets = list()

for sample in audio_samples:
    inp = Input(sample['input'], sample['target'], args.n_context)
    inputs.append(inp.prepare_net_inputs())
    targets.append(inp.prepare_net_targets())
    dims = inp.get_dimensions()
    n_input = dims['n_input']
    n_steps = dims['n_steps']

train_inputs = np.asarray(inputs)
train_targets = sparse_tuple_from(targets)

train_seq_len = [train_inputs.shape[1] for tis in xrange(batch_size)]
train_keep_prob = 1 - args.dropout_rate

# We don't have a validation dataset :(
val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len

val_keep_prob = train_keep_prob

dsNet = Network(n_steps, n_input, args.n_context, args.n_character, batch_size, args.relu_clip)
print("args.use_warpctc=", args.use_warpctc)
dsGraph, dsX, dsY, dsTargets, dsSeq_len, dsKeep_prob, dsLoss, dsCost, dsOptimizer, dsAccuracy, dsDecoded = dsNet.prepare(args.learning_rate, args.beta1, args.beta2, args.epsilon, args.use_warpctc)

print('"%s","%s","%s","%s","%s"' % ('run', 'epoch', 'cost', 'valerr', 'time'), file=args.csv)
for run in xrange(args.runs):
    print("Starting run #%d" % run, file=args.log)
    def print_status(curr, maxx, train_cost, train_ler, val_cost, val_ler, time):
        print('"%d","%d","%f","%f","%f"' % (run, curr, train_cost, val_ler, time), file=args.csv)
        if (curr % display_step == 0):
            log = "Epoch {}/{}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
            print(log.format(run+1, curr+1, maxx, train_cost, train_ler, val_cost, val_ler, time), file=args.log)

    t = Trainer(dsGraph, dsX, dsTargets, dsSeq_len, dsKeep_prob, dsCost, dsOptimizer, dsAccuracy, dsDecoded)
    str_decoded = t.train(args.training_iters, train_inputs, val_inputs, train_targets, val_targets, train_seq_len, val_seq_len, train_keep_prob, val_keep_prob, print_status)
    print("Decoded:\n%s" % str_decoded, file=args.log)

csvfile = args.csv.name
args.csv.close()

if args.plot:
    fdcsv = open(csvfile, 'r')
    Plotter(in_fname=fdcsv, out_fname=args.plot)
    fdcsv.close()
