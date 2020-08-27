import argparse
import importlib
import os
import re
import sys

from .helpers import secs_to_hours
from collections import Counter

def get_counter():
    return Counter({'all': 0, 'failed': 0, 'invalid_label': 0, 'too_short': 0, 'too_long': 0, 'imported_time': 0, 'total_time': 0})

def get_imported_samples(counter):
    return counter['all'] - counter['failed'] - counter['too_short'] - counter['too_long'] - counter['invalid_label']

def print_import_report(counter, sample_rate, max_secs):
    print('Imported %d samples.' % (get_imported_samples(counter)))
    if counter['failed'] > 0:
        print('Skipped %d samples that failed upon conversion.' % counter['failed'])
    if counter['invalid_label'] > 0:
        print('Skipped %d samples that failed on transcript validation.' % counter['invalid_label'])
    if counter['too_short'] > 0:
        print('Skipped %d samples that were too short to match the transcript.' % counter['too_short'])
    if counter['too_long'] > 0:
        print('Skipped %d samples that were longer than %d seconds.' % (counter['too_long'], max_secs))
    print('Final amount of imported audio: %s from %s.' % (secs_to_hours(counter['imported_time'] / sample_rate), secs_to_hours(counter['total_time'] / sample_rate)))

def get_importers_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--validate_label_locale', help='Path to a Python file defining a |validate_label| function for your locale. WARNING: THIS WILL ADD THIS FILE\'s DIRECTORY INTO PYTHONPATH.')
    return parser

def get_validate_label(args):
    """
    Expects an argparse.Namespace argument to search for validate_label_locale parameter.
    If found, this will modify Python's library search path and add the directory of the
    file pointed by the validate_label_locale argument.

    :param args: The importer's CLI argument object
    :type args: argparse.Namespace

    :return: The user-supplied validate_label function
    :type: function
    """
    # Python 3.5 does not support passing a pathlib.Path to os.path.* methods
    if 'validate_label_locale' not in args or (args.validate_label_locale is None):
        print('WARNING: No --validate_label_locale specified, your might end with inconsistent dataset.')
        return validate_label_eng
    validate_label_locale = str(args.validate_label_locale)
    if not os.path.exists(os.path.abspath(validate_label_locale)):
        print('ERROR: Inexistent --validate_label_locale specified. Please check.')
        return None
    module_dir = os.path.abspath(os.path.dirname(validate_label_locale))
    sys.path.insert(1, module_dir)
    fname = os.path.basename(validate_label_locale).replace('.py', '')
    locale_module = importlib.import_module(fname, package=None)
    return locale_module.validate_label

# Validate and normalize transcriptions. Returns a cleaned version of the label
# or None if it's invalid.
def validate_label_eng(label):
    # For now we can only handle [a-z ']
    if re.search(r"[0-9]|[(<\[\]&*{]", label) is not None:
        return None

    label = label.replace("-", " ")
    label = label.replace("_", " ")
    label = re.sub("[ ]{2,}", " ", label)
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace(";", "")
    label = label.replace("?", "")
    label = label.replace("!", "")
    label = label.replace(":", "")
    label = label.replace("\"", "")
    label = label.strip()
    label = label.lower()

    return label if label else None
