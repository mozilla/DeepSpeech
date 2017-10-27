#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import codecs
import fnmatch
import os
import subprocess
import unicodedata
import wave
import audioop
from random import shuffle
from shutil import copyfile

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
        self.add_command('help', self._cmd_help, 'Display help message')

    def add_command(self, name, action, description):
        cmd = _CommandLineParserCommand(name, action, description)
        self.commands[name] = cmd
        return cmd

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
        for cmd_name in self.commands:
            print()
            cmd = self.commands[cmd_name]
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

class Sample(object):
    def __init__(self, filename, transcript, filesize):
        self.filename = os.path.abspath(filename)
        self.transcript = transcript
        self.filesize = filesize
        self._duration = -1

    @property
    def duration(self):
        if self._duration < 0:
            self._duration = float(subprocess.check_output(['soxi', '-D', self.filename]))
        return self._duration

    def save_as(self, filename):
        filename = os.path.abspath(filename)
        copyfile(self.filename, filename)
        self.filename = filename

    def __str__(self):
        return '%s, %s, %f' % (self.filename, self.transcript, self.duration)


class DataSetBuilder(CommandLineParser):
    def __init__(self):
        super(DataSetBuilder, self).__init__()
        cmd = self.add_command('add', self._add, 'Adds samples listed in a CSV file to current buffer')
        cmd.add_argument('filename', 'string', 'Path to a CSV file')

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

        cmd = self.add_command('print', self._print, 'Prints list of samples in current buffer')

        cmd = self.add_command('play', self._play, 'Play samples of current buffer')

        cmd = self.add_command('write', self._write, 'Write samples of current buffer to disk')
        cmd.add_argument('dir_name', 'string', 'Path to the new sample directory. The directory and a file with the same name plus extension ".csv" should not exist.')

        cmd = self.add_command('augment', self._augment, 'Augment samples of current buffer with noise')

        self.samples = []

    def _add(self, filename):
        samples = [l.strip().split(',') for l in open(filename, 'r').readlines()[1:]]
        samples = [Sample(s[0], s[2], int(s[1])) for s in samples if len(s) == 3]
        self.samples.extend(samples)
        print('Added %d samples of CSV file "%s" to buffer.' % (len(samples), filename))

    def _shuffle(self):
        shuffle(self.samples)
        print('Shuffled buffer.')

    def _order(self):
        self.samples = sorted(self.samples, key=lambda s: s.filesize)
        print('Ordered buffer by file lenghts.')

    def _reverse(self):
        self.samples.reverse()
        print('Reversed buffer.')

    def _take(self, number):
        self.samples = self.samples[:number]
        print('Took %d samples as new buffer.' % number)

    def _repeat(self, number):
        self.samples = self.samples * number
        print('Repeated samples in buffer %d times as new buffer.' % number)

    def _skip(self, number):
        self.samples = self.samples[number:]
        print('Removed first %d samples from buffer.' % number)

    def _find(self, keyword):
        self.samples = [s for s in self.samples if keyword in s.transcript]
        print('Found %d samples containing keyword "%s".' % (len(self.samples), keyword))

    def _print(self):
        for s in self.samples:
            print(s)
        print('Printed %d samples.' % len(self.samples))

    def _play(self):
        for s in self.samples:
            print('Playing: ' + s.transcript)
            subprocess.call(['play', '-q', s.filename])
        print('Played %d samples.' % len(self.samples))

    def _write(self, dir_name):
        if dir_name[-1] == '/':
            dir_name = dir_name[:-1]
        csv_filename = dir_name + '.csv'
        if os.path.exists(dir_name) or os.path.exists(csv_filename):
            return 'Cannot write buffer, as either "%s" or "%s" already exist.' % (dir_name, csv_filename)
        os.makedirs(dir_name)
        for i, s in enumerate(self.samples):
            s.save_as('%s/sample-%d.wav' % (dir_name, i))
        with open(csv_filename, 'w') as csv:
            csv.write('wav_filename,wav_filesize,transcript\n')
            csv.write(''.join('%s,%d,%s\n' % (s.filename, s.filesize, s.transcript) for s in self.samples))
        print('Wrote %d samples to directory "%s" and listed them in CSV file "%s".' % (len(self.samples), dir_name, csv_filename))

    def _augment(self):
        print('Augment samples...')
        pass

def main():
    parser = DataSetBuilder()
    parser.parse(sys.argv[1:])

if __name__ == '__main__' :
    main()

