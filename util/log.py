from __future__ import print_function
import __builtin__
import sys

# used for looking up level from step=0 to error=4
LEVEL_NAMES = [
    'step',
    'debug',
    'info',
    'warn',
    'error'
]

def set_log_levels(log_level_str):
    '''
    Sets the global log levels by parsing a log level configuration string.
    Typically called at startup by main routine of a program (passing-in a command line value).
    'log_level_str' - comma separated assignments of log levels to modules (e.g. "main=debug,persistence=step")
        Modules are specified by their mudule IDs.
        Levels can be "step", "debug", "info", "warn" and "error".
        If a part specifies only a level, this level will be used as the default one.
    '''
    global DEFAULT_LOG_LEVEL
    global MODULE_LOG_LEVELS
    DEFAULT_LOG_LEVEL = 2 # default is info
    MODULE_LOG_LEVELS = {} # and no module specific levels
    for part in log_level_str.split(','):
        if part in LEVEL_NAMES:
            DEFAULT_LOG_LEVEL = LEVEL_NAMES.index(part)
        else:
            module_level = part.split('=')
            if len(module_level) == 2 and module_level[1] in LEVEL_NAMES:
                MODULE_LOG_LEVELS[module_level[0]] = LEVEL_NAMES.index(module_level[1])
    # persist settings in cross module global scope - other log-module imports will see those values
    __builtin__.DEFAULT_LOG_LEVEL = DEFAULT_LOG_LEVEL
    __builtin__.MODULE_LOG_LEVELS = MODULE_LOG_LEVELS

class Logger(object):
    '''
    Class for logging messages to the console. Supports log levels and module based logging.
    '''
    def __init__(self, id, caption=None):
        '''
        Constructs a new Logger instance.
        'id' - String identifier of the module this logger instance is created in.
            Should be simple, as it is used for parsing the log-level config string.
        'caption' - Full name of the module this logger instance is created in.
            Will be used as prefix in printed messages. If None (default), no prefix will be printed.
        '''
        self.id = id
        self.caption = caption

    def _print_message(self, msg_level, prefix, message, is_error=False):
        '''
        Internal method to print a (prefixed) log message in case the (module) log level allows this.
        'msg_level' - The log level (importance) of the message to print
        'prefix' - (One letter) prefix to indicate the type of this message to the user
        'message' - The actual message
        'is_error' - If the message should be printed to stderr instead of stdout.
        '''
        log_level = MODULE_LOG_LEVELS[self.id] if self.id in MODULE_LOG_LEVELS else DEFAULT_LOG_LEVEL
        if log_level <= msg_level:
            prefix = prefix + ' '
            if self.caption:
                prefix = prefix + '[' + self.caption + '] '
            text = prefix + ('\n' + prefix).join(message.split('\n'))
            print(text, file=sys.stdout if is_error else sys.stderr)

    def step(self, message):
        '''
        Prints a single step debug message for detailed debugging.
        '''
        self._print_message(0, 'S', str(message))

    def debug(self, message):
        '''
        Prints a debug message.
        '''
        self._print_message(1, 'D', str(message))

    def info(self, message):
        '''
        Prints an info message. Typically used for presenting results.
        '''
        self._print_message(2, 'I', str(message))

    def warn(self, message):
        '''
        Prints a warning. Typically used in case of resumable errors.
        '''
        self._print_message(3, 'W', str(message))

    def error(self, message):
        '''
        Prints an error message to stderr.
        '''
        self._print_message(4, 'E', str(message), is_error=True)
