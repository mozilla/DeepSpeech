#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import sox
import pandas
import random
from zipfile import ZipFile
import xml.etree.ElementTree as et
from util.text import validate_label

def _find_files(base_dir, extensions, name=None):
    for root, dirs, files in os.walk(base_dir):
        for basename in files:
            if name and not basename.startswith(name):
                continue
            for ext in extensions:
                if basename.endswith('.' + ext):
                    filename = os.path.join(root, basename)
                    yield filename
                    break

def _parse_time(time_string):
    components = time_string.split(':')
    components.reverse()
    power = 0
    seconds = 0
    for component in components:
        seconds += int(component) * (60 ** power)
        power += 1
    return seconds

def _xml_find_recursive(node, tag, ns=None):
    if ns != None:
        for n in ns:
            tag = tag.replace(n + ':', '{' + ns[n] + '}')

    for n in node.iter():
        if n.tag == tag:
            yield n
        _xml_find_recursive(n, tag, None)

def _process_xml(dataset, search_dir, data_dir, name, data):
    # Collect the annotation data
    ns = { 'vx': 'http://www.voxant.com/NewsML/transcript' }
    tree = et.fromstring(data)
    offset = 0
    snippets = []
    for turn_el in _xml_find_recursive(tree, 'vx:Turn', ns):
        speaker = turn_el.get('Speaker')

        if speaker == 'DISCLAIMER':
            for frag_el in turn_el.iterfind('vx:Fragment', ns):
                time_string = frag_el.get('StartTime')
                if time_string:
                    offset = _parse_time(time_string)
                    break
            continue

        for frag_el in turn_el.iterfind('vx:Fragment', ns):
            time_string = frag_el.get('StartTime')
            annotation = validate_label(frag_el.text)
            if time_string and annotation:
                snippets.append((_parse_time(time_string), annotation))

    if len(snippets) < 1:
        return

    # Recurse through the search path and look for audio files whose name
    # matches the given XML.
    for audiofile in _find_files(search_dir, ['wav', 'mp3', 'ogg', 'flac'], name):
        # First convert the entire audio into wav audio of the correct format
        tfm = sox.Transformer()
        tfm.convert(samplerate=16000, n_channels=1, bitdepth=16)
        convertedfile = os.path.join(data_dir,
                os.path.splitext(os.path.basename(audiofile))[0] + '.wav')
        tfm.build(audiofile, convertedfile)

        endtime = sox.file_info.duration(convertedfile) + offset
        snippets.append((endtime, ''))

        # Now split into multiple audio files per annotation
        nclips = 0
        for i in range(len(snippets) - 2):
            if snippets[i][0] >= snippets[i + 1][0]:
                continue

            tfm = sox.Transformer()
            tfm.trim(snippets[i][0] - offset, snippets[i + 1][0] - offset)
            snippetfile = os.path.splitext(convertedfile)[0] + '_' + str(snippets[i][0]) + '.wav'
            tfm.build(convertedfile, snippetfile)

            nclips += 1
            dataset.append((snippetfile, os.stat(snippetfile).st_size, snippets[i][1]))

        # Remove the unsplit audio file
        os.remove(convertedfile)

        print('Extracted %d clips from %s' % (nclips, name))
        return

def _search_and_preprocess_data(search_dir, data_dir):
    wav_dir = os.path.join(data_dir, 'audio')
    try:
        os.makedirs(wav_dir)
    except OSError:
        pass

    dataset = []

    # Recurse over directory tree and into zips looking for xml files
    for filename in _find_files(search_dir, ['xml', 'zip']):
        if filename.endswith('zip'):
            with ZipFile(filename, 'r') as xmlzip:
                for filenameinzip in xmlzip.namelist():
                    if filenameinzip.endswith('.xml'):
                        _process_xml(dataset, search_dir, wav_dir,
                                os.path.splitext(os.path.basename(filenameinzip))[0],
                                xmlzip.open(filenameinzip).read())
        else:
            f = open(filename, 'r')
            _process_xml(dataset, search_dir, wav_dir,
                    os.path.splitext(os.path.basename(filename))[0],
                    f.read())

    # Split data set into train/validation/test
    # We want an 90/10 random split between training+validation and test, then
    # the same between training and validation; so a 81/9/10 split.
    train = []
    validation = []
    test = []
    random.seed(0)
    for i in range(int(round(len(dataset) * 0.81))):
        train.append(dataset.pop(random.randint(0, len(dataset) - 1)))
    for i in range(int(round(len(dataset) * (9.0 / 19.0)))):
        validation.append(dataset.pop(random.randint(0, len(dataset) - 1)))
    for i in range(len(dataset)):
        test.append(dataset.pop(random.randint(0, len(dataset) - 1)))

    name = os.path.basename(data_dir)
    columns = ["wav_filename", "wav_filesize", "transcript"]
    pandas.DataFrame(data=train, columns=columns).to_csv(os.path.join(data_dir, name + '_train.csv'), index=False)
    validation_df = pandas.DataFrame(data=validation, columns=columns).to_csv(os.path.join(data_dir, name + '_validation.csv'), index=False)
    test_df = pandas.DataFrame(data=test, columns=columns).to_csv(os.path.join(data_dir, name + '_test.csv'), index=False)

if __name__ == "__main__":
    _search_and_preprocess_data(sys.argv[1], sys.argv[2])
