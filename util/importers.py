import argparse
import importlib
import os
import re
import sys

def get_importers_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--validate_label_locale', help='Path to a python module defining a |validate_label| function for your locale')
    return parser

def get_validate_label(args):
    if 'validate_label_locale' not in args or (args.validate_label_locale is None):
        print('WARNING: No --validate_label_locale specified, your might end with inconsistent dataset.')
        return validate_label_eng
    if not os.path.exists(os.path.abspath(args.validate_label_locale)):
        print('ERROR: Inexistent --validate_label_locale specified. Please check.')
        return None
    module_dir = os.path.abspath(os.path.dirname(args.validate_label_locale))
    sys.path.insert(1, module_dir)
    fname = os.path.basename(args.validate_label_locale).replace('.py', '')
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
