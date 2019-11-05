import sys
import os
import time
import logging
import traceback
import numpy as np
import wavTranscriber
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import shlex
import subprocess

# Debug helpers
logging.basicConfig(stream=sys.stderr,
                    level=logging.DEBUG,
                    format='%(filename)s - %(funcName)s@%(lineno)d %(name)s:%(levelname)s  %(message)s')


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:

    finished:
        No data

    error
       'tuple' (ecxtype, value, traceback.format_exc())

    result
        'object' data returned from processing, anything

    progress
            'object' indicating the transcribed result
    '''

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(object)


class Worker(QRunnable):
    '''
    Worker Thread

    Inherits from QRunnable to handle worker thread setup, signals and wrap-up

    @param callback:
    The funtion callback to run on this worker thread.
    Supplied args and kwargs will be passed through the runner.
    @type calllback: function
    @param args: Arguments to pass to the callback function
    @param kwargs: Keywords to pass to the callback function
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store the conctructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with the passed args, kwargs
        '''

        # Retrieve args/kwargs here; and fire up the processing using them
        try:
            transcript = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(transcript)
        finally:
            # Done
            self.signals.finished.emit()


class App(QMainWindow):
    dirName = ""

    def __init__(self):
        super().__init__()
        self.title = 'Deepspeech Transcriber'
        self.left = 10
        self.top = 10
        self.width = 480
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        layout = QGridLayout()
        layout.setSpacing(10)

        self.microphone = QRadioButton("Microphone")
        self.fileUpload = QRadioButton("File Upload")
        self.browseBox = QLineEdit(self, placeholderText="Wave File, Mono @ 16 kHz, 16bit Little-Endian")
        self.modelsBox = QLineEdit(self, placeholderText="Directory path for output_graph, lm & trie")
        self.textboxTranscript = QPlainTextEdit(self, placeholderText="Transcription")
        self.browseButton = QPushButton('Browse', self)
        self.browseButton.setToolTip('Select a wav file')
        self.modelsButton = QPushButton('Browse', self)
        self.modelsButton.setToolTip('Select deepspeech models folder')
        self.transcribeWav = QPushButton('Transcribe Wav', self)
        self.transcribeWav.setToolTip('Start Wav Transcription')
        self.openMicrophone = QPushButton('Start Speaking', self)
        self.openMicrophone.setToolTip('Open Microphone')

        layout.addWidget(self.microphone, 0, 1, 1, 2)
        layout.addWidget(self.fileUpload, 0, 3, 1, 2)
        layout.addWidget(self.browseBox, 1, 0, 1, 4)
        layout.addWidget(self.browseButton, 1, 4)
        layout.addWidget(self.modelsBox, 2, 0, 1, 4)
        layout.addWidget(self.modelsButton, 2, 4)
        layout.addWidget(self.transcribeWav, 3, 1, 1, 1)
        layout.addWidget(self.openMicrophone, 3, 3, 1, 1)
        layout.addWidget(self.textboxTranscript, 5, 0, -1, 0)

        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)

        # Microphone
        self.microphone.clicked.connect(self.mic_activate)

        # File Upload
        self.fileUpload.clicked.connect(self.wav_activate)

        # Connect Browse Button to Function on_click
        self.browseButton.clicked.connect(self.browse_on_click)

        # Connect the Models Button
        self.modelsButton.clicked.connect(self.models_on_click)

        # Connect Transcription button to threadpool
        self.transcribeWav.clicked.connect(self.transcriptionStart_on_click)

        # Connect Microphone button to threadpool
        self.openMicrophone.clicked.connect(self.openMicrophone_on_click)
        self.openMicrophone.setCheckable(True)
        self.openMicrophone.toggle()

        self.browseButton.setEnabled(False)
        self.browseBox.setEnabled(False)
        self.modelsBox.setEnabled(False)
        self.modelsButton.setEnabled(False)
        self.transcribeWav.setEnabled(False)
        self.openMicrophone.setEnabled(False)

        self.show()

        # Setup Threadpool
        self.threadpool = QThreadPool()
        logging.debug("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    @pyqtSlot()
    def mic_activate(self):
        logging.debug("Enable streaming widgets")
        self.en_mic = True
        self.browseButton.setEnabled(False)
        self.browseBox.setEnabled(False)
        self.modelsBox.setEnabled(True)
        self.modelsButton.setEnabled(True)
        self.transcribeWav.setEnabled(False)
        self.openMicrophone.setStyleSheet('QPushButton {background-color: #70cc7c; color: black;}')
        self.openMicrophone.setEnabled(True)

    @pyqtSlot()
    def wav_activate(self):
        logging.debug("Enable wav transcription widgets")
        self.en_mic = False
        self.openMicrophone.setStyleSheet('QPushButton {background-color: #f7f7f7; color: black;}')
        self.openMicrophone.setEnabled(False)
        self.browseButton.setEnabled(True)
        self.browseBox.setEnabled(True)
        self.modelsBox.setEnabled(True)
        self.modelsButton.setEnabled(True)

    @pyqtSlot()
    def browse_on_click(self):
        logging.debug('Browse button clicked')
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Select wav file to be Transcribed", "","All Files (*.wav)")
        if self.fileName:
            self.browseBox.setText(self.fileName)
            self.transcribeWav.setEnabled(True)
            logging.debug(self.fileName)

    @pyqtSlot()
    def models_on_click(self):
        logging.debug('Models Browse Button clicked')
        self.dirName = QFileDialog.getExistingDirectory(self, "Select deepspeech models directory")
        if self.dirName:
            self.modelsBox.setText(self.dirName)
            logging.debug(self.dirName)

            # Threaded signal passing worker functions
            worker = Worker(self.modelWorker, self.dirName)
            worker.signals.result.connect(self.modelResult)
            worker.signals.finished.connect(self.modelFinish)
            worker.signals.progress.connect(self.modelProgress)

            # Execute
            self.threadpool.start(worker)
        else:
            logging.critical("*****************************************************")
            logging.critical("Model path not specified..")
            logging.critical("*****************************************************")
            return "Transcription Failed, models path not specified"

    def modelWorker(self, dirName, progress_callback):
        self.textboxTranscript.setPlainText("Loading Models...")
        self.openMicrophone.setStyleSheet('QPushButton {background-color: #f7f7f7; color: black;}')
        self.openMicrophone.setEnabled(False)
        self.show()
        time.sleep(1)
        return dirName

    def modelProgress(self, s):
        # FixMe: Write code to show progress here
        pass

    def modelResult(self, dirName):
        # Fetch and Resolve all the paths of model files
        output_graph, lm, trie = wavTranscriber.resolve_models(dirName)
        # Load output_graph, alpahbet, lm and trie
        self.model = wavTranscriber.load_model(output_graph, lm, trie)

    def modelFinish(self):
        # self.timer.stop()
        self.textboxTranscript.setPlainText("Loaded Models, start transcribing")
        if self.en_mic is True:
            self.openMicrophone.setStyleSheet('QPushButton {background-color: #70cc7c; color: black;}')
            self.openMicrophone.setEnabled(True)
        self.show()

    @pyqtSlot()
    def transcriptionStart_on_click(self):
        logging.debug('Transcription Start button clicked')

        # Clear out older data
        self.textboxTranscript.setPlainText("")
        self.show()

        # Threaded signal passing worker functions
        worker = Worker(self.wavWorker, self.fileName)
        worker.signals.progress.connect(self.progress)
        worker.signals.result.connect(self.transcription)
        worker.signals.finished.connect(self.wavFinish)

        # Execute
        self.threadpool.start(worker)

    @pyqtSlot()
    def openMicrophone_on_click(self):
        logging.debug('Preparing to open microphone...')

        # Clear out older data
        self.textboxTranscript.setPlainText("")
        self.show()

        # Threaded signal passing worker functions
        # Prepare env for capturing from microphone and offload work to micWorker worker thread
        if (not self.openMicrophone.isChecked()):
            self.openMicrophone.setStyleSheet('QPushButton {background-color: #C60000; color: black;}')
            self.openMicrophone.setText("Stop")
            logging.debug("Start Recording pressed")
            logging.debug("Preparing for transcription...")

            sctx = self.model[0].createStream()
            subproc = subprocess.Popen(shlex.split('rec -q -V0 -e signed -L -c 1 -b 16 -r 16k -t raw - gain -2'),
                                       stdout=subprocess.PIPE,
                                       bufsize=0)
            self.textboxTranscript.insertPlainText('You can start speaking now\n\n')
            self.show()
            logging.debug('You can start speaking now')
            context = (sctx, subproc, self.model[0])

            # Pass the state to streaming worker
            worker = Worker(self.micWorker, context)
            worker.signals.progress.connect(self.progress)
            worker.signals.result.connect(self.transcription)
            worker.signals.finished.connect(self.micFinish)

            # Execute
            self.threadpool.start(worker)
        else:
            logging.debug("Stop Recording")

    '''
    Capture the audio stream from the microphone.
    The context is prepared by the openMicrophone_on_click()
    @param Context: Is a tuple containing three objects
                    1. Speech samples, sctx
                    2. subprocess handle
                    3. Deepspeech model object
    '''
    def micWorker(self, context, progress_callback):
        # Deepspeech Streaming will be run from this method
        logging.debug("Recording from your microphone")
        while (not self.openMicrophone.isChecked()):
            data = context[1].stdout.read(512)
            context[2].feedAudioContent(context[0], np.frombuffer(data, np.int16))
        else:
            transcript = context[2].finishStream(context[0])
            context[1].terminate()
            context[1].wait()
            self.show()
            progress_callback.emit(transcript)
            return "\n*********************\nTranscription Done..."

    def micFinish(self):
        self.openMicrophone.setText("Start Speaking")
        self.openMicrophone.setStyleSheet('QPushButton {background-color: #70cc7c; color: black;}')

    def transcription(self, out):
        logging.debug("%s" % out)
        self.textboxTranscript.insertPlainText(out)
        self.show()

    def wavFinish(self):
        logging.debug("File processed")

    def progress(self, chunk):
        logging.debug("Progress: %s" % chunk)
        self.textboxTranscript.insertPlainText(chunk)
        self.show()

    def wavWorker(self, waveFile, progress_callback):
        # Deepspeech will be run from this method
        logging.debug("Preparing for transcription...")
        inference_time = 0.0

        # Run VAD on the input file
        segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile, 1)
        f = open(waveFile.rstrip(".wav") + ".txt", 'w')
        logging.debug("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")

        for i, segment in enumerate(segments):
            # Run deepspeech on the chunk that just completed VAD
            logging.debug("Processing chunk %002d" % (i,))
            audio = np.frombuffer(segment, dtype=np.int16)
            output = wavTranscriber.stt(self.model[0], audio, sample_rate)
            inference_time += output[1]

            f.write(output[0] + " ")
            progress_callback.emit(output[0] + " ")

        # Summary of the files processed
        f.close()

        # Format pretty, extract filename from the full file path
        filename, ext = os.path.split(os.path.basename(waveFile))
        title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'LM Load Time(s)']
        logging.debug("************************************************************************************************************")
        logging.debug("%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
        logging.debug("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, self.model[1], self.model[2]))
        logging.debug("************************************************************************************************************")
        print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
        print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, self.model[1], self.model[2]))

        return "\n*********************\nTranscription Done..."


def main(args):
    app = QApplication(sys.argv)
    w = App()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv[1:])
