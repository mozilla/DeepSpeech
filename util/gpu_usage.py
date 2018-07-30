# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import os
import subprocess
import csv

from threading import Thread
from time import time
from scipy.interpolate import spline

from six.moves import range
# Do this to be able to use without X
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class GPUUsage(Thread):
    def __init__(self, csvfile=None):
        super(GPUUsage, self).__init__()

        self._cmd        = [ 'nvidia-smi', 'dmon', '-d', '1', '-s', 'pucvmet' ]
        self._names      = []
        self._units      = []
        self._process    = None

        self._csv_output = csvfile or os.environ.get('ds_gpu_usage_csv', self.make_basename(prefix='ds-gpu-usage', extension='csv'))

    def get_git_desc(self):
        return subprocess.check_output(['git', 'describe', '--always', '--abbrev']).strip()

    def make_basename(self, prefix, extension):
        # Let us assume that this code is executed in the current git clone
        return '%s.%s.%s.%s' % (prefix, self.get_git_desc(), int(time()), extension)

    def stop(self):
        if not self._process:
            print("Trying to stop nvidia-smi but no more process, please fix.")
            return

        print("Ending nvidia-smi monitoring: PID", self._process.pid)
        self._process.terminate()
        print("Ended nvidia-smi monitoring ...")

    def run(self):
        print("Starting nvidia-smi monitoring")

        # If the system has no CUDA setup, then this will fail.
        try:
            self._process = subprocess.Popen(self._cmd, stdout=subprocess.PIPE)
        except OSError as ex:
            print("Unable to start monitoring, check your environment:", ex)
            return

        writer = None
        with open(self._csv_output, 'w') as f:
            for line in iter(self._process.stdout.readline, ''):
                d = self.ingest(line)

                if line.startswith('# '):
                    if len(self._names) == 0:
                        self._names = d
                        writer = csv.DictWriter(f, delimiter=str(','), quotechar=str('"'), fieldnames=d)
                        writer.writeheader()
                        continue
                    if len(self._units) == 0:
                        self._units = d
                        continue
                else:
                    assert len(self._names) == len(self._units)
                    assert len(d) == len(self._names)
                    assert len(d) > 1
                    writer.writerow(self.merge_line(d))
                    f.flush()

    def ingest(self, line):
        return map(lambda x: x.replace('-', '0'), filter(lambda x: len(x) > 0, map(lambda x: x.strip(), line.split(' ')[1:])))

    def merge_line(self, line):
        return dict(zip(self._names, line))

class GPUUsageChart():
    def __init__(self, source, basename=None):
        self._rows    = [ 'pwr', 'temp', 'sm', 'mem']
        self._titles  = {
            'pwr':  "Power (W)",
            'temp': "Temperature (°C)",
            'sm':   "Streaming Multiprocessors (%)",
            'mem':  "Memory (%)"
        }
        self._data     = { }.fromkeys(self._rows)
        self._csv      = source
        self._basename = basename or os.environ.get('ds_gpu_usage_charts', 'gpu_usage_%%s_%d.png' % int(time.time()))

        # This should make sure we start from anything clean.
        plt.close("all")

        try:
            self.read()
            for plot in self._rows:
                self.produce_plot(plot)
        except IOError as ex:
            print("Unable to read", ex)

    def append_data(self, row):
        for bucket, value in row.iteritems():
            if not bucket in self._rows:
                continue

            if not self._data[bucket]:
                self._data[bucket] = {}

            gpu = int(row['gpu'])
            if not self._data[bucket].has_key(gpu):
                self._data[bucket][gpu]  = [ value ]
            else:
                self._data[bucket][gpu] += [ value ]

    def read(self):
        print("Reading data from", self._csv)
        with open(self._csv, 'r') as f:
            for r in csv.DictReader(f):
                self.append_data(r)

    def produce_plot(self, key, with_spline=True):
        png = self._basename % (key, )
        print("Producing plot for", key, "as", png)
        fig, axis = plt.subplots()
        data = self._data[key]
        if data is None:
            print("Data was empty, aborting")
            return

        x = list(range(len(data[0])))
        if with_spline:
            x = map(lambda x: float(x), x)
            x_sm = np.array(x)
            x_smooth = np.linspace(x_sm.min(), x_sm.max(), 300)

        for gpu, y in data.iteritems():
            if with_spline:
                y = map(lambda x: float(x), y)
                y_sm = np.array(y)
                y_smooth = spline(x, y, x_smooth, order=1)
                axis.plot(x_smooth, y_smooth, label='GPU %d' % (gpu))
            else:
                axis.plot(x, y, label='GPU %d' % (gpu))

        axis.legend(loc="upper right", frameon=False)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("%s" % self._titles[key])
        fig.set_size_inches(24, 18)
        plt.title("GPU Usage: %s" % self._titles[key])
        plt.savefig(png, dpi=100)
        plt.close(fig)
