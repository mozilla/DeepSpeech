import numpy as np
import time

import matplotlib.pyplot as plt
import csv

from natsort import natsorted

class Plotter:

    def __init__(self, in_fname, out_fname, plot_type):
        self._png = out_fname
        self._csv = in_fname

        self.load_csv()
        self.plot(plot_type)

    def load_csv(self):
        """ the csv file we read is made:
           'run', 'epoch', 'x', 'cost', 'valerr', 'time'
        """
        self._csv_content = {}
        maxRun  = 0
        maxIter = 0
        csv_file = csv.DictReader(self._csv, delimiter=',', quotechar='"')
        for line in csv_file:
            r = int(line['run'])
            try:
                self._csv_content[r].append(line)
            except KeyError as e:
                self._csv_content[r] = [ line ]
            maxRun  = max(maxRun, int(line['run']))
            maxIter = max(maxIter, int(line['epoch']))
        self._csv.close()

        self._X = xrange(maxIter + 1)
        self._C = np.ndarray(shape=(maxRun + 1, maxIter + 1), dtype=float)
        self._E = np.ndarray(shape=(maxRun + 1, maxIter + 1), dtype=float)
        self._T = np.ndarray(shape=(maxRun + 1, maxIter + 1), dtype=float)

        for run in xrange(maxRun+1):
            c = self._csv_content[run]
            self._C[run] = map(lambda x: x['cost'], c)
            self._E[run] = map(lambda x: x['valerr'], c)
            self._T[run] = map(lambda x: x['time'], c)

        self._meanC = np.mean(self._C, axis=0)
        self._ecC   = np.std(self._C, axis=0)

        self._meanE = np.mean(self._E, axis=0)
        self._ecE   = np.std(self._E, axis=0)

        self._meanT = np.mean(self._T, axis=0)
        self._ecT   = np.std(self._T, axis=0)

    def plot(self, type="loss"):
        if type == "loss":
            self.plot_loss()
        elif type == "valerr":
            self.plot_val_err()
	elif type == "time":
	    self.plot_time()
        ##else:
        ##    print("No plotting supported:", type)

    def plot_loss(self):
        fig, axis1 = plt.subplots()
        axis1.errorbar(self._X, self._meanC, yerr=self._ecC, color="red", label="loss")
        axis1.legend(loc="upper left", frameon=False)
        axis1.set_autoscaley_on(False)
        axis1.set_ylim([1e0, 1.1e3])

        axis2 = axis1.twinx()
        axis2.errorbar(self._X, self._meanT, yerr=self._ecT, color="blue", label="time")
        axis2.legend(loc="upper right", frameon=False)
        axis2.set_autoscaley_on(False)
        axis2.set_ylim([0, 2e1])

        axis1.set_xlabel("Epochs")
        axis1.set_yscale('log')

        fig.set_size_inches(24, 18)
        plt.title("Loss evolution")
        plt.savefig(self._png, dpi=100)

    def plot_val_err(self):
        fig, axis1 = plt.subplots()
        axis1.errorbar(self._X, self._meanE, yerr=self._ecE, color="red", label="valid. error")
        axis1.legend(loc="upper left", frameon=False)
        axis1.set_autoscaley_on(False)
        axis1.set_ylim([1e-5, 1e1])

        axis2 = axis1.twinx()
        axis2.errorbar(self._X, self._meanT, yerr=self._ecT, color="blue", label="time")
        axis2.legend(loc="upper right", frameon=False)
        axis2.set_autoscaley_on(False)
        axis2.set_ylim([0, 2e1])

        axis1.set_xlabel("Epochs")
        axis1.set_yscale('log')

        fig.set_size_inches(24, 18)
        plt.title("Validation error evolution")
        plt.savefig(self._png, dpi=100)

    def plot_time(self):
        fig, axis1 = plt.subplots()
        axis1.errorbar(self._X, self._meanT, yerr=self._ecT, color="blue", label="time")
        axis1.legend(loc="upper right", frameon=False)

        fig.set_size_inches(24, 18)
        plt.title("Execution time")
        plt.savefig(self._png, dpi=100)

class MultiPlotter(Plotter):

    def __init__(self, files, plot, type, title, xlbl, xtics, ylbl):
	self._files = natsorted(files, key=lambda y: y.name.lower())
        self._png   = plot
	self._title = title
	self._xlbl  = xlbl
	self._xtics = xtics
	self._ylbl  = ylbl

	self.load_csv()
	Plotter.plot(self, type)

    def load_csv(self):
        self._plotters = []
	for file in self._files:
	    self._plotters.append(Plotter(file, None, None))

        self._X = xrange(len(self._plotters))

	self._meanC = [ np.mean(X._C) for X in self._plotters ]
        self._ecC   = [ np.std(X._C) for X in  self._plotters ]

	self._meanE = [ np.mean(X._E) for X in self._plotters ]
        self._ecE   = [ np.std(X._E) for X in  self._plotters ]

	self._meanT = [ np.mean(X._T) for X in self._plotters ]
        self._ecT   = [ np.std(X._T) for X in  self._plotters ]

    def plot_time(self):
        fig, axis1 = plt.subplots()
        axis1.errorbar(self._X, self._meanT, yerr=self._ecT, color="blue", label="time")
        axis1.legend(loc="upper right", frameon=False)

	if self._xlbl:
            axis1.set_xlabel(self._xlbl)

        if self._xtics:
            axis1.set_xticks(self._xtics)

	if self._ylbl:
            axis1.set_ylabel(self._ylbl)

        fig.set_size_inches(24, 18)
	if self._title is None:
            plt.title("Execution time")
	else:
            plt.title(self._title)

        plt.savefig(self._png, dpi=100)
