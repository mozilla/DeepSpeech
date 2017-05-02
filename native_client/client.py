import sys
import scipy.io.wavfile as wav
from deepspeech import DeepSpeech

ds = DeepSpeech(sys.argv[1], 26, 9)
fs, audio = wav.read(sys.argv[2])
print ds.stt(audio, fs)
