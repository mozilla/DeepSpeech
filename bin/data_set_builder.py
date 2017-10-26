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
            cmd.action(*arg_values, **options)
        return None

    def parse(self, tokens):
        state = _CommandLineParserState(tokens)
        result = self._parse(state)
        if result:
            print(result)
            print()
            self._cmd_help()

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
    def __init__(self, filename, transcription, file_len):
        self.filename = filename
        self.transcription = transcription
        self.file_len = file_len
        self._duration = -1

    @property
    def duration(self):
        if self._duration < 0:
            self._duration = float(subprocess.check_output(['soxi', '-D', self.filename]))
        return self._duration

    def __str__(self):
        return '%s, %s, %f' % (self.filename, self.transcription, self.duration)


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

        cmd = self.add_command('print', self._print, 'Prints list of samples in current buffer')

        cmd = self.add_command('play', self._play, 'Play samples of current buffer')

        cmd = self.add_command('write', self._write, 'Write samples of current buffer to disk')
        cmd.add_argument('dir_name', 'int', 'Path to the new sample directory (should not exist)')

        cmd = self.add_command('augment', self._augment, 'Augment samples of current buffer with noise')

        self.samples = []

    def _add(self, filename):
        samples = [l.split(',') for l in open(filename, 'r').readlines()[1:]]
        self.samples.extend([Sample(s[0], s[2], int(s[1])) for s in samples])
        print('Added %d samples of CSV file "%s" to buffer.' % (len(samples), filename))

    def _shuffle(self):
        shuffle(self.samples)
        print('Shuffled buffer.')

    def _order(self):
        self.samples = sorted(self.samples, key=lambda s: s.file_len)
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

    def _print(self):
        for s in self.samples:
            print(s)
        print('Printed %d samples.' % len(self.samples))

    def _play(self):
        for s in self.samples:
            subprocess.call(['play', s.filename])
        print('Played %d samples.' % len(self.samples))

    def _write(self, dir_name):
        print('Write samples...')
        pass

    def _augment(self):
        print('Augment samples...')
        pass

def main():
    parser = DataSetBuilder()
    parser.parse(sys.argv[1:])

if __name__ == '__main__' :
    main()

