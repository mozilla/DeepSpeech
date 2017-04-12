#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath('.'))

from tensorflow.python.platform import gfile
from util.data_set_helpers import SwitchableDataSet, read_data_sets
import DeepSpeech as ds
import util.quantization as q

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def run_quantized_model(model=None):
    inputNames, outputNames = q.get_input_output_names()

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        with gfile.FastGFile(model, "rb") as src:
            gdef = tf.GraphDef()
            gdef.MergeFromString(src.read())
            input_output_nodes = tf.import_graph_def(gdef, return_elements=[ inputNames, outputNames ], name="")

        #print("Saved weights:", saved_weights
        print("Input/Output", input_output_nodes)

        g = tf.get_default_graph()

        # For dev purpose of quantization, we want to apply this process:
        # 1. Train a model on small corpus (e.g., TED, 10 samples, 300 epochs), dump that.
        # 2. Quantize the model
        # 3. Load quantized model, run it on the *train* data set
        #
        # Once trained on a real run of complete TED corpus, data_sets_quant.train should be
        # changed back to data_sets_quant.test to really exercise against Test set.
        print("Testing quantized model")

        # Set the global random seed for determinism
        tf.set_random_seed(FLAGS.random_seed)

        print("Reading dataset")

        # Get the required data set
        # TODO: Limit to 'test'
        data_sets = read_data_sets(FLAGS.train_files.split(','),
                                   FLAGS.dev_files.split(','),
                                   FLAGS.test_files.split(','),
                                   FLAGS.train_batch_size,
                                   FLAGS.dev_batch_size,
                                   FLAGS.test_batch_size,
                                   ds.n_input,
                                   ds.n_context,
                                   next_index=lambda set_name, index: ds.COORD.get_next_index(set_name),
                                   limit_dev=FLAGS.limit_dev,
                                   limit_test=FLAGS.limit_test,
                                   limit_train=FLAGS.limit_train)

        # Get the data sets
        data_set = SwitchableDataSet(data_sets)

        print("Creating quantization session")
        # Starting the execution context
        session_quant = tf.Session(config=ds.session_config, graph=g)

        print("Creating coordinator")
        # Create Coordinator to manage threads
        coord = tf.train.Coordinator()

        print("Starting queues")
        # Start queue runner threads
        managed_threads = tf.train.start_queue_runners(sess=session_quant, coord=coord)

        print("Starting threads")
        # Start importer's queue threads
        managed_threads = managed_threads + data_set.start_queue_threads(session_quant, coord)

        print("Calling do_inference")
        inference_time = ds.stopwatch()
        inference_total_loss, inference_avg_loss, inference_distance, inference_accuracy, inference_decoded, inference_labels = q.do_inference(batch_set=data_set, sess=session_quant, logits=input_output_nodes[1])
        inference_time = ds.stopwatch(inference_time)


        tower_labels = [ inference_labels ]
        tower_decodings = [ inference_decoded ]
        tower_distances = [ inference_distance ]
        tower_total_losses = [ inference_total_loss ]
        tower_accuracies = [ inference_accuracy ]
        tower_avg_losses = [ inference_avg_loss ]

        tower_gradients = []

        tower_results = (tower_labels, tower_decodings, tower_distances, tower_total_losses), \
                         tower_gradients, \
                         tf.reduce_mean(tower_accuracies, 0), \
                         tf.reduce_mean(tower_avg_losses, 0)

        #print("tower_accuracies=", tower_accuracies)
        #print("tower_results[tower_accuracies]=", tower_results[2])
        #print("tower_avg_losses=", tower_avg_losses)
        #print("tower_results[tower_avg_losses]=", tower_results[3])

        # inf_result = total_loss, avg_loss, distance, accuracy, decoded, batch_y
        # results_params, _, avg_accuracy, avg_loss = tower_results
        ## tower_results = (tower_labels, tower_decodings, tower_distances, tower_total_losses), \
        ##                  tower_gradients, \
        ##                  tf.reduce_mean(tower_accuracies, 0), \
        ##                  tf.reduce_mean(tower_avg_losses, 0)

        ##import pprint
        ##print("After inference:")
        ##print("tower_results=", len(tower_results))
        ##pprint.pprint(tower_results)
        ##for e in tower_results:
        ##    print("e=", e)

        ##    print("total_loss=", total_loss)
        ##    print("avg_loss=", avg_loss)
        ##    print("distance=", distance)
        ##    print("accuracy=", accuracy)
        ##    print("decoded=", decoded)
        ##    print("labels=", labels)

        ##for v in input_output_nodes:
        ##    print_one_var(session_quant, v)

        #print("Calling ds.calculate_loss_and_report")
        #import pprint

        # Applying the batches
        #result_quant = ds.calculate_loss_and_report(ctxt_quant, session_quant, epoch=-1, query_report=True)
        print("Computing accuracy and loss")
        batch_report, current_step, batch_accuracy, batch_loss = session_quant.run(tower_results, feed_dict={ data_set._queue_selector: 0 })
        #print("Result: result_quant=", len(result_quant), result_quant)
        #print("YOLO======")
        #pprint.pprint(yolo)

        #print("CURRENT_STEP======")
        #pprint.pprint(current_step)

        #print("BATCH_LOSS======")
        #pprint.pprint(batch_loss)

        #print("BATCH_ACCURACY======")
        #pprint.pprint(batch_accuracy)

        # Cleaning up
        #ds.stop_execution_context(ctxt_quant, session_quant, coord, managed_threads)

        #test_wer_quant = ds.print_report("Test Quantized", result_quant)

        print("Collecting results")
        report_results = ([],[],[],[])
        ds.collect_results(report_results, batch_report)
        wer, samples = ds.calculate_report(report_results)

        #pprint.pprint(report_results)
        #pprint.pprint(wer)
        #pprint.pprint(samples)

        test_epoch = ds.Epoch(index=0, num_jobs=0, set_name='test', report=True)
        test_epoch.accuracy = batch_accuracy 
        test_epoch.loss     = batch_loss
        test_epoch.wer      = wer
        test_epoch.samples.extend(samples)

        print(test_epoch)
        q.print_time(inference_time)

if __name__ == "__main__":
    tf.app.flags.DEFINE_string('run_quantized_model', '', 'Define which file to load to run Test WER')
    ds.initialize_globals()
    FLAGS.wer_log_file = ""
    run_quantized_model(os.path.abspath(FLAGS.run_quantized_model))
