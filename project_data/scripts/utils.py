"""
This script will store the general library functions
"""
import json
from config import *
import os

Speaker_Stats_Filename = "speaker_stats.json"


def list_all_files_with_ext(directory, ext):
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)) and file.endswith(ext):
            yield file


def dump_json(data, filename, parent_dir=Json_Home):
    complete_filename = os.path.join(parent_dir, filename)
    with open(complete_filename, 'w') as f:
        json.dump(data, f)


def load_json(filename, parent_dir=Json_Home):
    complete_filename = os.path.join(parent_dir, filename)
    with open(complete_filename) as f:
        return json.load(f)


def get_ids_for_speaker(speaker_id=Speaker_Id):
    return load_json(Speaker_Stats_Filename)[speaker_id]
