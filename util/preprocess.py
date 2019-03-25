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


def process_single_file(row, numcep, numcontext, alphabet):
    # row = index, Series
    _, file = row
    features = audiofile_to_input_vector(file.wav_filename, numcep, numcontext)
    features_len = len(features) - 2*numcontext
    transcript = np.frombuffer(file.transcript.encode('utf-8'), np.uint8).astype(np.int32)

    # if features_len < len(transcript):
    #     raise ValueError('Error: Audio file {} is too short for transcription.'.format(file.wav_filename))

    return features, features_len, transcript, len(transcript)


# load samples from CSV, compute features, optionally cache results on disk
def preprocess(csv_files, batch_size, numcep, numcontext, alphabet, hdf5_cache_path=None):
    COLUMNS = ('features', 'features_len', 'transcript', 'transcript_len')

    print('Preprocessing', csv_files)

    if hdf5_cache_path and os.path.exists(hdf5_cache_path):
        with tables.open_file(hdf5_cache_path, 'r') as file:
            features = file.root.features[:]
            features_len = file.root.features_len[:]
            transcript = file.root.transcript[:]
            transcript_len = file.root.transcript_len[:]

            # features are stored flattened, so reshape into [n_steps, numcep]
            for i in range(len(features)):
                features[i].shape = [features_len[i]+2*numcontext, numcep]

            in_data = list(zip(features, features_len,
                               transcript, transcript_len))
            print('Loaded from cache at', hdf5_cache_path)
            return pandas.DataFrame(data=in_data, columns=COLUMNS)

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
                      alphabet=alphabet)
    out_data = pmap(step_fn, source_data.iterrows())

    if hdf5_cache_path:
        print('Saving to', hdf5_cache_path)

        # list of tuples -> tuple of lists
        features, features_len, transcript, transcript_len = zip(*out_data)

        with tables.open_file(hdf5_cache_path, 'w') as file:
            features_dset = file.create_vlarray(file.root,
                                                'features',
                                                tables.Float32Atom(),
                                                filters=tables.Filters(complevel=1))
            # VLArray atoms need to be 1D, so flatten feature array
            for f in features:
                features_dset.append(np.reshape(f, -1))

            features_len_dset = file.create_array(file.root,
                                                  'features_len',
                                                  features_len)

            transcript_dset = file.create_vlarray(file.root,
                                                  'transcript',
                                                  tables.Int32Atom(),
                                                  filters=tables.Filters(complevel=1))
            for t in transcript:
                transcript_dset.append(t)

            transcript_len_dset = file.create_array(file.root,
                                                    'transcript_len',
                                                    transcript_len)

    print('Preprocessing done')
    return pandas.DataFrame(data=out_data, columns=COLUMNS)
