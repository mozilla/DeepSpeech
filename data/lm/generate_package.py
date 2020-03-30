#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

import argparse
import shutil

from util.text import Alphabet, UTF8Alphabet
from ds_ctcdecoder import Scorer, Alphabet as NativeAlphabet


def create_bundle(
    alphabet_path,
    lm_path,
    vocab_path,
    package_path,
    force_utf8,
    default_alpha,
    default_beta,
):
    words = set()
    vocab_looks_char_based = True
    with open(vocab_path) as fin:
        for line in fin:
            for word in line.split():
                words.add(word.encode("utf-8"))
                if len(word) > 1:
                    vocab_looks_char_based = False
    print("{} unique words read from vocabulary file.".format(len(words)))
    print(
        "{} like a character based model.".format(
            "Looks" if vocab_looks_char_based else "Doesn't look"
        )
    )

    if force_utf8 != None:  # pylint: disable=singleton-comparison
        use_utf8 = force_utf8.value
        print("Forcing UTF-8 mode = {}".format(use_utf8))
    else:
        use_utf8 = vocab_looks_char_based

    if use_utf8:
        serialized_alphabet = UTF8Alphabet().serialize()
    else:
        if not alphabet_path:
            print("No --alphabet path specified, can't continue.")
            sys.exit(1)
        serialized_alphabet = Alphabet(alphabet_path).serialize()

    alphabet = NativeAlphabet()
    err = alphabet.deserialize(serialized_alphabet, len(serialized_alphabet))
    if err != 0:
        print("Error loading alphabet: {}".format(err))
        sys.exit(1)

    scorer = Scorer()
    scorer.set_alphabet(alphabet)
    scorer.set_utf8_mode(use_utf8)
    scorer.reset_params(default_alpha, default_beta)
    scorer.load_lm(lm_path)
    scorer.fill_dictionary(list(words))
    shutil.copy(lm_path, package_path)
    scorer.save_dictionary(package_path, True)  # append, not overwrite
    print("Package created in {}".format(package_path))


class Tristate(object):
    def __init__(self, value=None):
        if any(value is v for v in (True, False, None)):
            self.value = value
        else:
            raise ValueError("Tristate value must be True, False, or None")

    def __eq__(self, other):
        return (
            self.value is other.value
            if isinstance(other, Tristate)
            else self.value is other
        )

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        raise TypeError("Tristate object may not be used as a Boolean")

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return "Tristate(%s)" % self.value


def main():
    parser = argparse.ArgumentParser(
        description="Generate an external scorer package for DeepSpeech."
    )
    parser.add_argument(
        "--alphabet",
        help="Path of alphabet file to use for vocabulary construction. Words with characters not in the alphabet will not be included in the vocabulary. Optional if using UTF-8 mode.",
    )
    parser.add_argument(
        "--lm",
        required=True,
        help="Path of KenLM binary LM file. Must be built without including the vocabulary (use the -v flag). See generate_lm.py for how to create a binary LM.",
    )
    parser.add_argument(
        "--vocab",
        required=True,
        help="Path of vocabulary file. Must contain words separated by whitespace.",
    )
    parser.add_argument("--package", required=True, help="Path to save scorer package.")
    parser.add_argument(
        "--default_alpha",
        type=float,
        required=True,
        help="Default value of alpha hyperparameter.",
    )
    parser.add_argument(
        "--default_beta",
        type=float,
        required=True,
        help="Default value of beta hyperparameter.",
    )
    parser.add_argument(
        "--force_utf8",
        default="",
        help="Boolean flag, force set or unset UTF-8 mode in the scorer package. If not set, infers from the vocabulary.",
    )
    args = parser.parse_args()

    if args.force_utf8 in ("True", "1", "true", "yes", "y"):
        force_utf8 = Tristate(True)
    elif args.force_utf8 in ("False", "0", "false", "no", "n"):
        force_utf8 = Tristate(False)
    else:
        force_utf8 = Tristate(None)

    create_bundle(
        args.alphabet,
        args.lm,
        args.vocab,
        args.package,
        force_utf8,
        args.default_alpha,
        args.default_beta,
    )


if __name__ == "__main__":
    main()
