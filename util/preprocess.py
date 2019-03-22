import numpy as np
import os
import pandas
import tables

from functools import partial
from multiprocessing.dummy import Pool
from util.audio import audiofile_to_input_vector
from util.text import text_to_char_array

def pmap(fun, iterable):
    pool = Pool()
    results = pool.map(fun, iterable)
    pool.close()
    return results


def process_single_file(row, numcep, numcontext, alphabet, file_type):
    # row = index, Series
    _, file = row
    features = audiofile_to_input_vector(file.wav_filename, numcep, numcontext)
    features_len = len(features) - 2*numcontext
    transcript = np.array([0] if file_type == 'kyrgyz' else [1], dtype=np.float32)

    return features, features_len, transcript, 1


# load samples from CSV, compute features, optionally cache results on disk
def preprocess(csv_files, batch_size, numcep, numcontext, alphabet, file_type, hdf5_cache_path=None):
    COLUMNS = ('features', 'features_len', 'transcript', 'transcript_len')

    print('Preprocessing', csv_files)

    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1)))
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file)

    step_fn = partial(process_single_file,
                      numcep=numcep,
                      numcontext=numcontext,
                      alphabet=alphabet,
                      file_type=file_type)
    out_data = pmap(step_fn, source_data.iterrows())

    print('Preprocessing done')
    return pandas.DataFrame(data=out_data, columns=COLUMNS)
