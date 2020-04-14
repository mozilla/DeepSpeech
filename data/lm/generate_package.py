#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import argparse
import shutil

from deepspeech_training.util.text import Alphabet, UTF8Alphabet
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

    cbm = "Looks" if vocab_looks_char_based else "Doesn't look"
    print("{} like a character based model.".format(cbm))

    if force_utf8 in ("True", "1", "true", "yes", "y"):
        use_utf8 = True
    elif force_utf8 in ("False", "0", "false", "no", "n"):
        use_utf8 = False
    else:
        use_utf8 = vocab_looks_char_based
        print("Using detected UTF-8 mode: {}".format(use_utf8))

    if use_utf8:
        serialized_alphabet = UTF8Alphabet().serialize()
    else:
        if not alphabet_path:
            raise RuntimeError("No --alphabet path specified, can't continue.")
        serialized_alphabet = Alphabet(alphabet_path).serialize()

    alphabet = NativeAlphabet()
    err = alphabet.deserialize(serialized_alphabet, len(serialized_alphabet))
    if err != 0:
        raise RuntimeError("Error loading alphabet: {}".format(err))

    scorer = Scorer()
    scorer.set_alphabet(alphabet)
    scorer.set_utf8_mode(use_utf8)
    scorer.reset_params(default_alpha, default_beta)
    scorer.load_lm(lm_path)
    scorer.fill_dictionary(list(words))
    shutil.copy(lm_path, package_path)
    scorer.save_dictionary(package_path, True)  # append, not overwrite
    print("Package created in {}".format(package_path))


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
        help="Boolean flag, force set or unset UTF-8 mode in the scorer package. If not set, infers from the vocabulary. See <https://github.com/mozilla/DeepSpeech/blob/master/doc/Decoder.rst#utf-8-mode> for further explanation",
    )
    args = parser.parse_args()

    create_bundle(
        args.alphabet,
        args.lm,
        args.vocab,
        args.package,
        args.force_utf8,
        args.default_alpha,
        args.default_beta,
    )


if __name__ == "__main__":
    main()
