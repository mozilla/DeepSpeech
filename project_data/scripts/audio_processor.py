"""
This script is responsible for transforming and pre-processing flac to wav files.
The deepspeech tool only accepts 16-bit 16Khz 1-channel wav files as input
Please run the following command to generate the durations of each of the audio files prior to using this scipt:
    $ sh scripts/get_flac_audio_length.sh | tee ~/data/intermediate_results/flac_duration.log
"""

from config import *
from utils import *
import os

Audio_Window = 10
Audio_Stride = 8


def convert_audio_ffmpeg(src_filename, target_filename, start_point, duration):
    cmd = "ffmpeg -ss %d -t %d -y -i %s %s" % (start_point, duration, src_filename, target_filename)
    print("Executing: %s" % cmd)
    os.system(cmd)


def split_files(duration_log_filename, duration, stride):
    """
    This method calls for the convert_audio_ffmpeg method on all the flac files in the directory
    """
    required_fileids = get_ids_for_speaker()
    max_duration = 0
    with open(duration_log_filename) as mf:
        for line in mf.read().splitlines():
            complete_input_filename, length_str = line.split(":")
            particular_file_id = complete_input_filename.split(os.sep)[-1].split(Flac_Ext)[0]
            if particular_file_id in required_fileids:
                length = float(length_str)
                iteration = 0
                start = 0
                if length > max_duration:
                    max_duration = length
                while True:
                    target_filename = os.path.join(Audio_Corpora_Home, "%s_%d.wav" % (particular_file_id, iteration))
                    convert_audio_ffmpeg(complete_input_filename, target_filename, start, duration)
                    if length - start < duration:
                        break
                    else:
                        start = start + stride
                        iteration = iteration + 1
        print("The maximum duration file handled is: %f" % max_duration)    # In our case it's 16.4 seconds


if __name__ == "__main__":
    duration_log = "%s/intermediate_results/flac_duration.log" % Global_Root
    split_files(duration_log, Audio_Window, Audio_Stride)
