from DeepSpeech import Plotter

import sys
import argparse

parser = argparse.ArgumentParser(
  description='Some basic DeepSpeech plotter',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
  '--csv', type=argparse.FileType('r'),
  required=True, help='Where to read computed data from')

parser.add_argument(
  '--plot', type=str, default=None,
  required=True, help='File name where to plot resulting learning of the network')

parser.add_argument(
  '--plot-type', type=str, choices=['loss', 'valerr'],
  default='loss', help='Type of data to plot')

args = parser.parse_args()

Plotter(in_fname=args.csv, out_fname=args.plot, plot_type=args.plot_type)
