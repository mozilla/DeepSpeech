#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with deepspeech model.

The client queries the service with the given audio and prints a ranked list
of decoded outputs to the standard output, one per line.

Typical usage example:

    deepspeech_client.py --server=localhost:9000 --file audio.wav
"""

import os
import sys

local_tf = os.path.join(os.path.dirname(os.path.dirname(os.path.join(os.path.abspath(__file__)))), 'local_tf')
sys.path.append(local_tf)

import threading
from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from util.text import ndarray_to_text
from util.audio import audiofile_to_input_vector

tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('file', '', 'Wave audio file')
# These need to match the constants used when training the deepspeech model
tf.app.flags.DEFINE_string('n_input', 26, 'Number of MFCC features')
tf.app.flags.DEFINE_string('n_context', 9, 'Number of frames of context')
FLAGS = tf.app.flags.FLAGS

def _create_rpc_callback(event):
    def _callback(result_future):
        exception = result_future.exception()
        if exception:
            print exception
        else:
            results = tf.contrib.util.make_ndarray(result_future.result().outputs['outputs'])
            for result in results[0]:
                print ndarray_to_text(result)
        event.set()
    return _callback

def do_inference(hostport, audio):
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'deepspeech'
    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(audio))

    event = threading.Event()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(_create_rpc_callback(event))
    if event.is_set() != True:
        event.wait()

def main(_):
    if not FLAGS.server:
        print 'please specify server host:port'
        return
    if not FLAGS.file:
        print 'pleace specify an audio file'
        return

    audio_waves = audiofile_to_input_vector(
                  FLAGS.file, FLAGS.n_input, FLAGS.n_context)
    audio = np.array([ audio_waves ])
    do_inference(FLAGS.server, audio)

if __name__ == '__main__':
    tf.app.run()
