"""
This script is used for generating the speaker-specific stats
"""

from config import *
from utils import dump_json
from collections import defaultdict
import os


def get_speaker(filename):
    assert filename.endswith(Speaker_Ext)
    with open(filename) as f:
        line = f.read().splitlines()[0]
        return line.split()[0]


def get_speaker_id_map():
    speaker_id_map = defaultdict(list)
    for filename in os.listdir(Librispeech_Home):
        if filename.endswith(Speaker_Ext):
            speaker = get_speaker(os.path.join(Librispeech_Home, filename))
            speaker_id_map[speaker].append(filename.split(Speaker_Ext)[0])
    return speaker_id_map


if __name__ == '__main__':
    speaker_stats = get_speaker_id_map()
    dump_json(speaker_stats, Speaker_Stats_Filename)
    speaker_max_files = max(speaker_stats, key=(lambda speaker: len(speaker_stats[speaker])))
    print("The speaker with max files: %s and num_files=%d" %
          (speaker_max_files, len(speaker_stats[speaker_max_files])))
