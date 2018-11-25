from __future__ import print_function

from util.flags import FLAGS


# Logging functions
# =================

def prefix_print(prefix, message):
    print(prefix + ('\n' + prefix).join(message.split('\n')))


def log_debug(message):
    if FLAGS.log_level == 0:
        prefix_print('D ', message)


def log_traffic(message):
    if FLAGS.log_traffic:
        log_debug(message)


def log_info(message):
    if FLAGS.log_level <= 1:
        prefix_print('I ', message)


def log_warn(message):
    if FLAGS.log_level <= 2:
        prefix_print('W ', message)


def log_error(message):
    if FLAGS.log_level <= 3:
        prefix_print('E ', message)