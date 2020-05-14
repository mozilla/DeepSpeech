#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import absl.app
import argparse
import numpy as np
import wave
import csv
import os
import sys

from deepspeech import Model
from deepspeech_training.util.evaluate_tools import calculate_and_print_report
from deepspeech_training.util.flags import create_flags
from functools import partial
from multiprocessing import JoinableQueue, Process, cpu_count, Manager
from six.moves import zip, range

r'''
This module should be self-contained:
  - build libdeepspeech.so with TFLite:
    - bazel build [...] --define=runtime=tflite [...] //native_client:libdeepspeech.so
  - make -C native_client/python/ TFDIR=... bindings
  - setup a virtualenv
  - pip install native_client/python/dist/deepspeech*.whl
  - pip install -r requirements_eval_tflite.txt

Then run with a TF Lite model, a scorer and a CSV test file
'''

def tflite_worker(model, scorer, queue_in, queue_out, gpu_mask):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_mask)
    ds = Model(model)
    ds.enableExternalScorer(scorer)

    while True:
        try:
            msg = queue_in.get()

            filename = msg['filename']
            fin = wave.open(filename, 'rb')
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
            fin.close()

            decoded = ds.stt(audio)

            queue_out.put({'wav': filename, 'prediction': decoded, 'ground_truth': msg['transcript']})
        except FileNotFoundError as ex:
            print('FileNotFoundError: ', ex)

        print(queue_out.qsize(), end='\r') # Update the current progress
        queue_in.task_done()

def main(args, _):
    manager = Manager()
    work_todo = JoinableQueue()   # this is where we are going to store input data
    work_done = manager.Queue()  # this where we are gonna push them out

    processes = []
    for i in range(args.proc):
        worker_process = Process(target=tflite_worker, args=(args.model, args.scorer, work_todo, work_done, i), daemon=True, name='tflite_process_{}'.format(i))
        worker_process.start()        # Launch reader() as a separate python process
        processes.append(worker_process)

    print([x.name for x in processes])

    wavlist = []
    ground_truths = []
    predictions = []
    losses = []
    wav_filenames = []

    with open(args.csv, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        count = 0
        for row in csvreader:
            count += 1
            # Relative paths are relative to the folder the CSV file is in
            if not os.path.isabs(row['wav_filename']):
                row['wav_filename'] = os.path.join(os.path.dirname(args.csv), row['wav_filename'])
            work_todo.put({'filename': row['wav_filename'], 'transcript': row['transcript']})
            wav_filenames.extend(row['wav_filename'])

    print('Totally %d wav entries found in csv\n' % count)
    work_todo.join()
    print('\nTotally %d wav file transcripted' % work_done.qsize())

    while not work_done.empty():
        msg = work_done.get()
        losses.append(0.0)
        ground_truths.append(msg['ground_truth'])
        predictions.append(msg['prediction'])
        wavlist.append(msg['wav'])

    # Print test summary
    _ = calculate_and_print_report(wav_filenames, ground_truths, predictions, losses, args.csv)

    if args.dump:
        with open(args.dump + '.txt', 'w') as ftxt, open(args.dump + '.out', 'w') as fout:
            for wav, txt, out in zip(wavlist, ground_truths, predictions):
                ftxt.write('%s %s\n' % (wav, txt))
                fout.write('%s %s\n' % (wav, out))
            print('Reference texts dumped to %s.txt' % args.dump)
            print('Transcription   dumped to %s.out' % args.dump)

def parse_args():
    parser = argparse.ArgumentParser(description='Computing TFLite accuracy')
    parser.add_argument('--model', required=True,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=True,
                        help='Path to the external scorer file')
    parser.add_argument('--csv', required=True,
                        help='Path to the CSV source file')
    parser.add_argument('--proc', required=False, default=cpu_count(), type=int,
                        help='Number of processes to spawn, defaulting to number of CPUs')
    parser.add_argument('--dump', required=False,
                        help='Path to dump the results as text file, with one line for each wav: "wav transcription".')
    args, unknown = parser.parse_known_args()
    # Reconstruct argv for absl.flags
    sys.argv = [sys.argv[0]] + unknown
    return args

if __name__ == '__main__':
    create_flags()
    absl.app.run(partial(main, parse_args()))
