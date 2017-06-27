import sys
import scipy.io.wavfile as wav
from deepspeech.model import Model

ds = Model(sys.argv[1], 26, 9)
fs, audio = wav.read(sys.argv[2])
print ds.stt(audio, fs)
