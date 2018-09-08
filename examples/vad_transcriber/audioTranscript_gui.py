import sys
import os
import inspect
import logging
import traceback
import numpy as np
import wavTranscriber
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Debug helpers
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def PrintFrame():
    # 0 represents this line
    # 1 represents line at caller
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    logging.debug(info.function, info.lineno)


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
        self.height = 320
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        layout = QGridLayout()
        layout.setSpacing(10)

        self.textbox = QLineEdit(self, placeholderText="Wave File, Mono @ 16 kHz, 16bit Little-Endian")
        self.modelsBox = QLineEdit(self, placeholderText="Directory path for output_graph, alphabet, lm & trie")
        self.textboxTranscript = QPlainTextEdit(self, placeholderText="Transcription")
        self.button = QPushButton('Browse', self)
        self.button.setToolTip('Select a wav file')
        self.modelsButton = QPushButton('Browse', self)
        self.modelsButton.setToolTip('Select deepspeech models folder')
        self.transcribeButton = QPushButton('Transcribe', self)
        self.transcribeButton.setToolTip('Start Transcription')

        layout.addWidget(self.textbox, 0, 0)
        layout.addWidget(self.button, 0, 1)
        layout.addWidget(self.modelsBox, 1, 0)
        layout.addWidget(self.modelsButton, 1, 1)
        layout.addWidget(self.transcribeButton, 2, 0, Qt.AlignHCenter)
        layout.addWidget(self.textboxTranscript, 3, 0, -1, 0)

        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)

        # Connect Button to Function on_click
        self.button.clicked.connect(self.on_click)

        # Connect the Models Button
        self.modelsButton.clicked.connect(self.models_on_click)

        # Connect Transcription button to threadpool
        self.transcribeButton.clicked.connect(self.transcriptionStart_on_click)
        self.show()

        # Setup Threadpool
        self.threadpool = QThreadPool()
        logging.debug("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    @pyqtSlot()
    def on_click(self):
        logging.debug('Browse button clicked')
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self,"Select wav file to be Transcribed", \
                            "","All Files (*.wav)", options=options)
        if self.fileName:
            self.textbox.setText(self.fileName)
            logging.debug(self.fileName)

    @pyqtSlot()
    def models_on_click(self):
        logging.debug('Models Browse Button clicked')
        self.dirName = QFileDialog.getExistingDirectory(self,"Select deepspeech models directory")
        if self.dirName:
            self.modelsBox.setText(self.dirName)
            logging.debug(self.dirName)

    @pyqtSlot()
    def transcriptionStart_on_click(self):
        logging.debug('Transcription Start button clicked')

        # Clear out older data
        self.textboxTranscript.setPlainText("")
        self.show()

        # Threaded signal passing worker functions
        worker = Worker(self.runDeepspeech, self.fileName)
        worker.signals.result.connect(self.transcription)
        worker.signals.finished.connect(self.threadComplete)
        worker.signals.progress.connect(self.progress)

        # Execute
        self.threadpool.start(worker)

    def transcription(self, out):
        logging.debug("Transcribed text: %s" % out)
        self.textboxTranscript.insertPlainText(out)
        self.show()

    def threadComplete(self):
        logging.debug("File processed")

    def progress(self, chunk):
        logging.debug("Progress: %s" % chunk)
        self.textboxTranscript.insertPlainText(chunk)
        self.show()

    def runDeepspeech(self, waveFile, progress_callback):
        # Deepspeech will be run from this method
        logging.debug("Preparing for transcription...")

        # Go and fetch the models from the directory specified
        if self.dirName:
            # Resolve all the paths of model files
            output_graph, alphabet, lm, trie = wavTranscriber.resolve_models(self.dirName)
        else:
            logging.critical("*****************************************************")
            logging.critical("Model path not specified..")
            logging.critical("You sure of what you're doing ?? ")
            logging.critical("Trying to fetch from present working directory.")
            logging.critical("*****************************************************")
            return "Transcription Failed, models path not specified"

        # Load output_graph, alpahbet, lm and trie
        model_retval = wavTranscriber.load_model(output_graph, alphabet, lm, trie)
        inference_time = 0.0

        # Run VAD on the input file
        segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile, 1)
        f = open(waveFile.rstrip(".wav") + ".txt", 'w')
        logging.debug("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")

        for i, segment in enumerate(segments):
            # Run deepspeech on the chunk that just completed VAD
            logging.debug("Processing chunk %002d" % (i,))
            audio = np.frombuffer(segment, dtype=np.int16)
            output = wavTranscriber.stt(model_retval[0], audio, sample_rate)
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
        logging.debug("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
        logging.debug("************************************************************************************************************")
        print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
        print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))

        return "\n*********************\nTranscription Done..."


def main(args):
    app = QApplication(sys.argv)
    w = App()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv[1:])
