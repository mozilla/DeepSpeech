#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import math
import codecs
import fnmatch
import os
import subprocess
import unicodedata
import wave
import audioop
import tempfile
import glob
from threading import Lock
from random import shuffle
from shutil import copyfile
from intervaltree import IntervalTree
from pydub import AudioSegment
from multiprocessing.dummy import Pool

class Error(Exception):
    def __init__(self, message):
        self.message = message

class _CommandLineParserCommand(object):
    def __init__(self, name, action, description):
        self.name = name
        self.action = action
        self.description = description
        self.arguments = []
        self.options = {}
    def add_argument(self, name, type, description):
        assert type != 'bool'
        self.arguments.append(_CommandLineParserParameter(name, type, description))
    def add_option(self, name, type, description):
        self.options[name] = _CommandLineParserParameter(name, type, description)

class _CommandLineParserParameter(object):
    def __init__(self, name, type, description):
        self.name = name
        self.type = type
        self.description = description

class _CommandLineParserState(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = -1
    @property
    def token(self):
        return self.tokens[self.index]
    def next(self):
        self.index += 1
        return self.index < len(self.tokens)
    def prev(self):
        self.index -= 1
        return self.index >= 0


class CommandLineParser(object):
    def __init__(self):
        self.commands = {}
        self.command_list = []
        self.add_command('help', self._cmd_help, 'Display help message')

    def add_command(self, name, action, description):
        cmd = _CommandLineParserCommand(name, action, description)
        self.commands[name] = cmd
        self.command_list.append(cmd)
        return cmd

    def add_group(self, caption):
        self.command_list.append(caption)

    def _parse_value(self, state, value_type):
        if value_type == 'bool':
            return True
        if not state.next():
            return None
        try:
            if value_type == 'int':
                return int(state.token)
            if value_type == 'float':
                return float(state.token)
        except:
            state.prev()
            return None
        return state.token

    def _parse(self, state):
        while state.next():
            if not state.token in self.commands:
                return "Unrecognized command: %s" % state.token
            cmd = self.commands[state.token]
            arg_values = []
            for arg in cmd.arguments:
                arg_value = self._parse_value(state, arg.type)
                if not arg_value:
                    return "Problem parsing argument %s of command %s" % (arg.name, cmd.name)
                arg_values.append(arg_value)
            options = {}
            while state.next() and state.token[0] == '-':
                opt_name = state.token[1:]
                if not opt_name in cmd.options:
                    return "Unknown option -%s for command %s" % (opt_name, cmd.name)
                opt = cmd.options[opt_name]
                opt_value = self._parse_value(state, opt.type)
                if opt_value == None:
                    return "Unable to parse %s value for option -%s of command %s" % (opt.type, opt.name, cmd.name)
                options[opt_name] = opt_value
            state.prev()
            result = cmd.action(*arg_values, **options)
            if result:
                return result
        return None

    def parse(self, tokens):
        state = _CommandLineParserState(tokens)
        result = self._parse(state)
        if result:
            print(result)
            print()
            self._cmd_help()
            return

    def _cmd_help(self):
        print('Usage: import_fisher.py (command <arg1> <arg2> ... [-opt1 [<value>]] [-opt2 [<value>]] ...)*')
        print('Commands:')
        for cmd in self.command_list:
            print()
            if isinstance(cmd, basestring):
                print(cmd + ':')
                continue
            arg_desc = ' '.join('<%s>' % arg.name for arg in cmd.arguments)
            opt_desc = ' '.join(('[-%s%s]' % (opt.name, ' <%s>' % opt.name if opt.type != 'bool' else '')) for _, opt in cmd.options.items())
            print('  %s %s %s' % (cmd.name, arg_desc, opt_desc))
            print('\t%s' % cmd.description)
            if len(cmd.arguments) > 0:
                print('\tArguments:')
                for arg in cmd.arguments:
                    print('\t\t%s: %s - %s' % (arg.name, arg.type, arg.description))
            if len(cmd.options) > 0:
                print('\tOptions:')
                for _, opt in cmd.options.items():
                    print('\t\t-%s: %s - %s' % (opt.name, opt.type, opt.description))

tmp_dir = None
tmp_index = 0
tmp_lock = Lock()

def get_tmp_filename():
    global tmp_index, tmp_dir
    with tmp_lock:
        if not tmp_dir:
            tmp_dir = tempfile.mkdtemp()
        tmp_index += 1
        return '%s/%d.wav' % (tmp_dir, tmp_index)

class WavFile(object):
    def __init__(self, filename=None):
        self.filename = os.path.abspath(filename) if filename else get_tmp_filename()
        self.file_is_tmp = not filename
        self._stats = None
        self._duration = -1
        self._filesize = -1

    def __del__(self):
        if self.file_is_tmp:
            os.remove(self.filename)

    def save_as(self, filename):
        filename = os.path.abspath(filename)
        if self.file_is_tmp:
            os.rename(self.filename, filename)
            self.filename = filename
            self.file_is_tmp = False
            return self
        file = WavFile(filename=filename)
        copyfile(self.filename, file.filename)
        file._stats = self._stats
        file._duration = self._duration
        file._filesize = self._filesize
        return file

    @property
    def stats(self):
        self.write()
        if not self._stats:
            entries = subprocess.check_output(['soxi', self.filename], stderr=subprocess.STDOUT)
            entries = entries.strip().split('\n')
            entries = [e.split(':')[:2] for e in entries]
            entries = [(e[0].strip(), e[1].strip()) for e in entries if len(e) == 2]
            self._stats = { key: value for (key, value) in entries }
        return self._stats

    @property
    def duration(self):
        if self._duration < 0:
            self._duration = float(subprocess.check_output(['soxi', '-D', self.filename]).strip())
        return self._duration

    @property
    def filesize(self):
        if self._filesize < 0:
            self._filesize = os.path.getsize(self.filename)
        return self._filesize

    @property
    def volume(self):
        return float(self.stats['Volume adjustment'])

class Sample(object):
    def __init__(self, file, transcript):
        self.file = file
        self.transcript = transcript
        self.effects = ''

    def write(self, filename=None):
        if len(self.effects) > 0:
            file = WavFile(filename=filename)
            subprocess.call(['sox', self.file.filename, file.filename] + self.effects.strip().split(' '))
            self.effects = ''
            self.file = file
        elif filename:
            self.file = self.file.save_as(filename)

    def add_sox_effect(self, effect):
        self.effects += ' %s' % effect

    def read_audio_segment(self):
        self.write()
        return AudioSegment.from_file(self.file.filename, format="wav")

    def write_audio_segment(self, segment):
        self.file = WavFile()
        segment.export(self.file.filename, format="wav")

    def clone(self):
        sample = Sample(self.file, self.transcript)
        sample.effects = self.effects
        return sample

    def __str__(self):
        return 'Filename: %s\nTranscript: %s' % (self.file.filename, self.transcript)

class DataSetBuilder(CommandLineParser):
    def __init__(self):
        super(DataSetBuilder, self).__init__()
        cmd = self.add_command('add', self._add, 'Adds samples listed in named buffer or a file to current buffer')
        cmd.add_argument('source', 'string', 'Name of a named buffer or filename of a CSV file or WAV file (wildcards supported)')

        self.add_group('Buffer operations')

        cmd = self.add_command('shuffle', self._shuffle, 'Randoimize order of the sample buffer')

        cmd = self.add_command('order', self._order, 'Order samples in buffer by length')

        cmd = self.add_command('reverse',self._reverse, 'Reverse order of samples in buffer')

        cmd = self.add_command('take', self._take, 'Take given number of samples from the beginning of the buffer as new buffer')
        cmd.add_argument('number', 'int', 'Number of samples')

        cmd = self.add_command('repeat', self._repeat, 'Repeat samples of current buffer <number> times as new buffer')
        cmd.add_argument('number', 'int', 'How often samples of the buffer should get repeated')

        cmd = self.add_command('skip', self._skip, 'Skip given number of samples from the beginning of current buffer')
        cmd.add_argument('number', 'int', 'Number of samples')

        cmd = self.add_command('find', self._find, 'Drop all samples, who\'s transcription does not contain a keyword' )
        cmd.add_argument('keyword', 'string', 'Keyword to look for in transcriptions')

        cmd = self.add_command('clear', self._clear, 'Clears sample buffer')

        self.add_group('Named buffers')

        cmd = self.add_command('set', self._set, 'Replaces named buffer with contents of buffer')
        cmd.add_argument('name', 'string', 'Name of the named buffer')

        cmd = self.add_command('stash', self._stash, 'Moves buffer to named buffer (buffer will be empty afterwards)')
        cmd.add_argument('name', 'string', 'Name of the named buffer')

        cmd = self.add_command('push', self._push, 'Appends buffer to named buffer')
        cmd.add_argument('name', 'string', 'Name of the named buffer')

        cmd = self.add_command('drop', self._drop, 'Drops named buffer')
        cmd.add_argument('name', 'string', 'Name of the named buffer')

        self.add_group('Output')

        cmd = self.add_command('print', self._print, 'Prints list of samples in current buffer')

        cmd = self.add_command('play', self._play, 'Play samples of current buffer')

        cmd = self.add_command('write', self._write, 'Write samples of current buffer to disk')
        cmd.add_argument('dir_name', 'string', 'Path to the new sample directory. The directory and a file with the same name plus extension ".csv" should not exist.')

        self.add_group('Effects')

        # wet_only=False, reverberance=0.5, hf_damping=0.5, room_scale=1.0, stereo_depth=1.0, pre_delay=0, wet_gain=0
        cmd = self.add_command('reverb', self._reverb, 'Adds reverberation to buffer samples')
        cmd.add_option('wet_only', 'bool', 'If to strip source signal on output')
        cmd.add_option('reverberance', 'float', 'Reverberance factor (between 0.0 to 1.0)')
        cmd.add_option('hf_damping', 'float', 'HF damping factor (between 0.0 to 1.0)')
        cmd.add_option('room_scale', 'float', 'Room scale factor (between 0.0 to 1.0)')
        cmd.add_option('stereo_depth', 'float', 'Stereo depth factor (between 0.0 to 1.0)')
        cmd.add_option('pre_delay', 'int', 'Pre delay in ms')
        cmd.add_option('wet_gain', 'float', 'Wet gain in dB')

        cmd = self.add_command('echo', self._echo, 'Adds an echo effect to buffer samples')
        cmd.add_argument('gain_in', 'float', 'Gain in')
        cmd.add_argument('gain_out', 'float', 'Gain out')
        cmd.add_argument('delay_decay', 'string', 'Comma separated delay decay pairs - at least one (e.g. 10,0.1,20,0.2)')

        cmd = self.add_command('speed', self._speed, 'Adds an speed effect to buffer samples')
        cmd.add_argument('factor', 'float', 'Speed factor to apply')

        cmd = self.add_command('sox', self._sox, 'Adds a SoX effect to buffer samples')
        cmd.add_argument('effect', 'string', 'SoX effect name')
        cmd.add_argument('args', 'string', 'Comma separated list of SoX effect parameters (no white space allowed)')

        cmd = self.add_command('augment', self._augment, 'Augment samples of current buffer with noise')
        cmd.add_argument('source', 'string', 'CSV file with samples to augment onto current sample buffer')
        cmd.add_option('times', 'int', 'How often to apply the augmentation source to the sample buffer')
        cmd.add_option('gain', 'float', 'How much gain (in dB) to apply to augmentation audio before overlaying onto buffer samples')

        self.named_buffers = {}
        self.samples = []

    def _map(self, lst, fun, threads=8):
        pool = Pool(threads)
        results = pool.map(fun, lst)
        pool.close()
        pool.join()
        return results

    def _clone_buffer(self, buffer):
        samples = []
        for sample in buffer:
            samples.append(sample.clone())
        return samples

    def _load_samples(self, source):
        ext = source[-4:].lower()
        samples = []
        if ext == '.csv':
            samples = [l.strip().split(',') for l in open(source, 'r').readlines()[1:]]
            samples = [Sample(WavFile(filename=s[0]), s[2]) for s in samples if len(s) == 3]
        elif ext == '.wav':
            samples = glob.glob(source)
            samples = [Sample(WavFile(filename=s), '') for s in samples]
        elif source in self.named_buffers:
            samples = self._clone_buffer(self.named_buffers[source])
        if len(samples) == 0:
            raise Error('No samples found!')
        return samples

    def _add(self, source):
        samples = self._load_samples(source)
        self.samples.extend(samples)
        print('Added %d samples to buffer.' % len(samples))

    def _shuffle(self):
        shuffle(self.samples)
        print('Shuffled buffer.')

    def _order(self):
        self.samples = sorted(self.samples, key=lambda s: s.file.filesize)
        print('Ordered buffer by file lenghts.')

    def _reverse(self):
        self.samples.reverse()
        print('Reversed buffer.')

    def _take(self, number):
        self.samples = self.samples[:number]
        print('Took %d samples as new buffer.' % number)

    def _repeat(self, number):
        samples = self.samples[:]
        for _ in range(number - 1):
            for sample in self.samples:
                samples.append(sample.clone())
        self.samples = samples
        print('Repeated samples in buffer %d times as new buffer.' % number)

    def _skip(self, number):
        self.samples = self.samples[number:]
        print('Removed first %d samples from buffer.' % number)

    def _clear(self):
        self.samples = []
        print('Removed all samples from buffer.')

    def _set(self, name):
        self.named_buffers[name] = self._clone_buffer(self.samples)
        print('Set named buffer "%s" to samples of buffer.' % name)

    def _stash(self, name):
        self.named_buffers[name] = self.samples
        self.samples = []
        print('Set named buffer "%s" to samples of buffer.' % name)

    def _push(self, name):
        if not name in self.named_buffers:
            self.named_buffers[name] = []
        self.named_buffers[name].extend(self._clone_buffer(self.samples))
        print('Appended samples of buffer to named buffer "%s".' % name)

    def _drop(self, name):
        del self.named_buffers[name]
        print('Dropped named buffer "%s".' % name)

    def _find(self, keyword):
        self.samples = [s for s in self.samples if keyword in s.transcript]
        print('Found %d samples containing keyword "%s".' % (len(self.samples), keyword))

    def _print(self):
        for s in self.samples:
            print(s)
        print('Printed %d samples.' % len(self.samples))

    def _play(self):
        for s in self.samples:
            s.write()
            print('Playing: ' + s.transcript)
            subprocess.call(['play', '-q', s.file.filename])
        print('Played %d samples.' % len(self.samples))

    def _write(self, dir_name):
        if dir_name[-1] == '/':
            dir_name = dir_name[:-1]
        csv_filename = dir_name + '.csv'
        if os.path.exists(dir_name) or os.path.exists(csv_filename):
            return 'Cannot write buffer, as either "%s" or "%s" already exist.' % (dir_name, csv_filename)
        os.makedirs(dir_name)
        samples = [(i, sample) for i, sample in enumerate(self.samples)]
        def write_sample(i_sample):
            i, sample = i_sample
            sample.write(filename='%s/sample-%d.wav' % (dir_name, i))
        self._map(samples, write_sample)
        with open(csv_filename, 'w') as csv:
            csv.write('wav_filename,wav_filesize,transcript\n')
            csv.write(''.join('%s,%d,%s\n' % (s.file.filename, s.file.filesize, s.transcript) for s in self.samples))
        print('Wrote %d samples to directory "%s" and listed them in CSV file "%s".' % (len(self.samples), dir_name, csv_filename))

    def _reverb(self, wet_only=False, reverberance=0.5, hf_damping=0.5, room_scale=1.0, stereo_depth=1.0, pre_delay=0, wet_gain=0):
        effect = 'reverb %s%d %d %d %d %d %d' % \
            ('-w ' if wet_only else '', int(reverberance*100.0), int(hf_damping*100.0), int(room_scale*100.0), int(stereo_depth*100.0), pre_delay, wet_gain)
        for s in self.samples:
            s.add_sox_effect(effect)
        print('Added reverberation to %d samples in buffer.' % len(self.samples))

    def _echo(self, gain_in, gain_out, delay_decay):
        delay_decay = delay_decay.split(',')
        assert len(delay_decay) % 2 == 0
        assert len(delay_decay) > 1
        effect = 'echo %f %f %s' % (gain_in, gain_out, ' '.join(delay_decay))
        for s in self.samples:
            s.add_sox_effect(effect)
        print('Added echo effect to %d samples in buffer.' % len(self.samples))

    def _speed(self, factor):
        effect = 'speed %f' % factor
        for s in self.samples:
            s.add_sox_effect(effect)
        print('Added speed effect to %d samples in buffer.' % len(self.samples))

    def _sox(self, effect, args):
        effect = '%s %s' % (effect, ' '.join(args.split(',')))
        for s in self.samples:
            s.add_sox_effect(effect)
        print('Added %s effect to %d samples in buffer.' % (effect, len(self.samples)))

    def _augment(self, source, times=1, gain=-8):
        aug_samples = self._load_samples(source)
        tree = IntervalTree()

        aug_durs = self._map(aug_samples, lambda s: int(math.ceil(s.file.duration * 1000.0)))
        total_aug_dur = sum(aug_durs)
        position = 0
        for i, sample in enumerate(aug_samples):
            duration = aug_durs[i]
            tree[position:position+duration] = sample
            position += duration

        def prepare_sample(s):
            s.write()
            return int(math.ceil(s.file.duration * 1000.0))
        orig_durs = self._map(self.samples, prepare_sample)
        total_orig_dur = sum(orig_durs)

        positions = []
        position = 0
        for i, sample in enumerate(self.samples):
            duration = orig_durs[i]
            positions.append((position, sample))
            position += duration

        def augment_sample(pos_sample):
            position, sample = pos_sample
            orig_seg = sample.read_audio_segment()
            orig_dur = len(orig_seg)
            aug_seg = AudioSegment.silent(duration=orig_dur)
            sub_pos = position
            for i in range(times):
                inters = tree[sub_pos:sub_pos + orig_dur]
                for inter in inters:
                    seg = inter.data.read_audio_segment()
                    offset = inter.begin - sub_pos
                    if offset < 0:
                        seg = seg[-offset:]
                        offset = 0
                    aug_seg = aug_seg.overlay(seg, position=offset)
                sub_pos = (sub_pos + total_orig_dur) % total_aug_dur
            aug_seg = aug_seg + (orig_seg.dBFS - aug_seg.dBFS + gain)
            orig_seg = orig_seg.overlay(aug_seg)
            sample.write_audio_segment(orig_seg)

        self._map(positions, augment_sample)
        print('Augmented %d samples in buffer.' % len(self.samples))

def main():
    parser = DataSetBuilder()
    parser.parse(sys.argv[1:])

if __name__ == '__main__' :
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted by user')
    #except Exception as ex:
    #    print(ex)