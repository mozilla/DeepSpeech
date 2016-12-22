#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with deepspeech model.

The client launches a local web server, by default accessible at localhost:8080
that provides a user interface to demonstrations of the DeepSpeech model.

Typical usage example:

    deepspeech_demos.py --server=localhost:9000
"""

import os
import sys
import time
import wave
import base64
import signal
import socket
import hashlib
import pyaudio
import StringIO
import threading
import webrtcvad
import numpy as np
import SocketServer
import SimpleHTTPServer
import tensorflow as tf
from array import array
from grpc.beta import implementations
from websocket_server import WebsocketServer
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from util.text import ndarray_to_text
from util.audio import audiofile_to_input_vector

tf.app.flags.DEFINE_integer('port', 8080, 'PredictionService host:port')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
# These need to match the constants used when training the deepspeech model
tf.app.flags.DEFINE_integer('n_input', 26, 'Number of MFCC features')
tf.app.flags.DEFINE_integer('n_context', 9, 'Number of frames of context')
tf.app.flags.DEFINE_integer('stride', 2, 'CTC stride on time axis')
FLAGS = tf.app.flags.FLAGS

FRAME_SIZE = 160
FRAME_LENGTH = 10
SILENCE_BEFORE_COMPLETE = (500 * FRAME_SIZE) / FRAME_LENGTH
MAXIMUM_LENGTH = (10000 * FRAME_SIZE) / FRAME_LENGTH
SEND_INTERVAL = (300 * FRAME_SIZE) / FRAME_LENGTH

def _create_rpc_callback(event, server):
    def _callback(result_future):
        exception = result_future.exception()
        if exception:
            print exception
        else:
            results = tf.contrib.util.make_ndarray(result_future.result().outputs['outputs'])
            for result in results[0]:
                server.message(ndarray_to_text(result))
        event.set()
    return _callback

def do_inference(hostport, audio_file, server):
    audio_waves = audiofile_to_input_vector(
                  audio_file, FLAGS.n_input, FLAGS.n_context, FLAGS.stride)
    audio = np.array([ audio_waves ])

    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'deepspeech'
    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(audio))

    event = threading.Event()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(_create_rpc_callback(event, server))
    if event.is_set() != True:
        event.wait()

class NoCacheHTTPRequestHAndler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_headers()
        SimpleHTTPServer.SimpleHTTPRequestHandler.end_headers(self)

    def send_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')

class DSSocketServer(threading.Thread):
    def __init__(self, port=9876):
        threading.Thread.__init__(self)
        self.setDaemon = True

        self.server = WebsocketServer(port)
        self.server.set_fn_message_received(self.message_received)

        self.stopped = 0

    def message_received(self, client, server, message):
        print 'Message received: %s' % (message)
        if message == 'STOP':
            self.stopped += 1
        elif message == 'START':
            self.stopped -= 1

    def run(self):
        self.server.run_forever()

    def message(self, text):
        self.server.send_message_to_all(text)

class DSWebServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.setDaemon = True

        self.server = SocketServer.ThreadingTCPServer(('', FLAGS.port),
                                                      NoCacheHTTPRequestHAndler)
        self.server.allow_reuse_address = True

        self.socket_server = DSSocketServer()

    def run(self):
        print 'Starting server, visit http://localhost:%d/' % (FLAGS.port)
        self.socket_server.start()
        self.server.serve_forever()

    def message(self, text):
        self.socket_server.message(text)

    @property
    def stopped(self):
        return self.socket_server.stopped > 0

def main(_):
    if not FLAGS.server:
        print 'please specify server host:port'
        return

    vad = webrtcvad.Vad()
    pa = pyaudio.PyAudio()

    # Default to using pulse
    device = 0
    for i in xrange(0, pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['name'] == 'pulse':
            device = i
            break

    # 320 frames = 10ms @ 16-bit 16kHz
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                     input_device_index=device, input=True,
                     #output_device_index=device, output=True, # for debugging
                     frames_per_buffer=FRAME_SIZE)

    silent_frames = 0
    recorded = StringIO.StringIO()
    recorded = array('h')
    recording = False

    # Start web server
    server = DSWebServer()
    server.start()

    # Start recording/transcribing/serving
    print('Listening...')
    while True:
        while stream.is_active() and stream.get_read_available() >= FRAME_SIZE:
            audio = array('h', stream.read(FRAME_SIZE))
            if sys.byteorder == 'big':
                audio.byteswap()

            if vad.is_speech(audio.tostring(), 16000):
                if recording != True:
                    recording = True
                    print 'Recording...'
                    server.message('RECORD')
                silent_frames = 0
            else:
                if recording:
                    silent_frames += 1

            if recording:
                recorded.extend(audio)

            if len(recorded) >= MAXIMUM_LENGTH:
                break

        if server.stopped == stream.is_active():
            if server.stopped:
                stream.stop_stream()
                if len(recorded):
                    recorded = array('h')
                recording = False
                silent_frames = 0
                print 'Stopped recording'
            else:
                stream.start_stream()
                print 'Resume recording'

        if recording and len(recorded) % SEND_INTERVAL is 0:
            audiofile = StringIO.StringIO()
            encoder = wave.open(audiofile, 'wb')
            encoder.setnchannels(1)
            encoder.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            encoder.setframerate(16000)
            encoder.writeframes(recorded.tostring())
            encoder.close()

            # For debugging
            #stream.write(audiofile.getvalue())

            audiofile.seek(0)
            #sys.stdout.write('\033[2J\033[H') # Clear screen, return to home
            do_inference(FLAGS.server, audiofile, server)
            audiofile.close()

            if silent_frames >= SILENCE_BEFORE_COMPLETE or len(recorded) >= MAXIMUM_LENGTH:
                server.message('END')
                print('Listening...')
                silent_frames = 0
                recorded = array('h')
                recording = False

    stream.stop_stream()
    stream.close()
    pa.terminate()

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        os._exit(0)
