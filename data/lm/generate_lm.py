import argparse
import gzip
import io
import os
import subprocess
from collections import Counter
from itertools import zip_longest
import shutil

import progressbar


def convert_and_filter_topk(args):
    """ Convert to lowercase, count word occurrences and save top-k words to a file """

    # Loop over multiple input_txt arguments
    data_lowers = []
    vocabs = []
    vocab_str_combined = ""
    fill = args.top_k[-1] # with multiple input_txt arguments where there are fewer top_k
                          # arguments, the last top_k will recur for the extra input_txts
    for input_txt_num, (input_txt_item, top_k_item) in \
            enumerate(zip_longest(args.input_txt,args.top_k, fillvalue=fill), 1):

        print(f"\nProcessing input_txt {input_txt_num} of {len(args.input_txt)}")
        print(f"  {input_txt_item}")
        counter = Counter()
        data_lower = os.path.join(args.output_dir, f"lower_{input_txt_num}.txt.gz")
        data_lowers.append(data_lower)

        print("\nConverting to lowercase and counting word occurrences ...")
        with io.TextIOWrapper(
            io.BufferedWriter(gzip.open(data_lower, "w+")), encoding="utf-8"
        ) as file_out:

            # Open the input file either from input.txt or input.txt.gz
            _, file_extension = os.path.splitext(input_txt_item)
            if file_extension == ".gz":
                file_in = io.TextIOWrapper(
                    io.BufferedReader(gzip.open(input_txt_item)), encoding="utf-8"
                )
            else:
                file_in = open(input_txt_item, encoding="utf-8")

            for line in progressbar.progressbar(file_in):
                line_lower = line.lower()
                counter.update(line_lower.split())
                file_out.write(line_lower)

            file_in.close()

        # Save top-k words
        print(f"\nSaving top {top_k_item} words ...")
        top_counter = counter.most_common(top_k_item)
        vocab_str = "\n".join(word for word, count in top_counter)
        vocab_str_combined = vocab_str_combined + '\n' + vocab_str
        vocab_path = f"vocab_{input_txt_num}-{top_k_item}.txt"
        vocab_path = os.path.join(args.output_dir, vocab_path)
        vocabs.append(vocab_path)
        with open(vocab_path, "w+") as file:
            file.write(vocab_str)

        print(f"\nCalculating word statistics for input_txt {input_txt_num}...")
        total_words = sum(counter.values())
        print(f"  Your text file has {total_words} words in total")
        print(f"  It has {len(counter)} unique words")
        top_words_sum = sum(count for word, count in top_counter)
        word_fraction = (top_words_sum / total_words) * 100
        print(f"  Your top-{top_k_item} words are {word_fraction:.4f}"
              f" percent of all words")
        first_word, first_count = top_counter[0]
        last_word, last_count = top_counter[-1]
        print(f'  Your most common word "{first_word}" occurred {first_count} times')
        print(f'  The least common word in your top-k is "{last_word}" with {last_count} times')
        for i, (w, c) in enumerate(reversed(top_counter)):
            if c > last_count:
                print(f'  The first word with {c} occurrences is "{w}"'
                      f' at place {len(top_counter) - 1 - i}')
                break

    data_lower = os.path.join(args.output_dir, "lower.txt.gz")
    # Combine the multiple input_txt iterations
    with open(data_lower,"wb") as dl:
        for f in data_lowers:
            with open(f,"rb") as fd:
                shutil.copyfileobj(fd, dl)
            os.remove(f)

    # Combine the multiple vocab iterations
    vocab_path = os.path.join(args.output_dir, "vocab_combined.txt")
    with open(vocab_path,"wb") as dl:
        for f in vocabs:
            with open(f,"rb") as fd:
                shutil.copyfileobj(fd, dl)
            os.remove(f)

    return data_lower, vocab_str_combined


def build_lm(args, data_lower, vocab_str):
    print("\nCreating ARPA file ...")
    lm_path = os.path.join(args.output_dir, "lm.arpa")
    subargs = [
            os.path.join(args.kenlm_bins, "lmplz"),
            "--order",
            str(args.arpa_order),
            "--temp_prefix",
            args.output_dir,
            "--memory",
            args.max_arpa_memory,
            "--text",
            data_lower,
            "--arpa",
            lm_path,
            "--prune",
            *args.arpa_prune.split("|"),
        ]
    if args.discount_fallback:
        subargs += ["--discount_fallback"]
    subprocess.check_call(subargs)

    # Filter LM using vocabulary of top-k words
    print("\nFiltering ARPA file using vocabulary of top-k words ...")
    filtered_path = os.path.join(args.output_dir, "lm_filtered.arpa")
    subprocess.run(
        [
            os.path.join(args.kenlm_bins, "filter"),
            "single",
            f"model:{lm_path}",
            filtered_path,
        ],
        input=vocab_str.encode("utf-8"),
        check=True,
    )

    # Quantize and produce trie binary.
    print("\nBuilding lm.binary ...")
    binary_path = os.path.join(args.output_dir, "lm.binary")
    subprocess.check_call(
        [
            os.path.join(args.kenlm_bins, "build_binary"),
            "-a",
            str(args.binary_a_bits),
            "-q",
            str(args.binary_q_bits),
            "-v",
            args.binary_type,
            filtered_path,
            binary_path,
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate lm.binary and top-k vocab for DeepSpeech."
    )
    parser.add_argument(
        "--input_txt", "-i",
        help="Path to a file.txt or file.txt.gz with sample sentences. "
             "Pass argument multiple times for multiple files.",
        type=str,
        action='append',
        required=True,
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Directory path for the output",
        type=str,
        required=True
    )
    parser.add_argument(
        "--top_k", "-k",
        help="Use top_k most frequent words for the vocab.txt file. "
             "These will be used to filter the ARPA file. "
             "Optionally pass argument multiple times for multiple input_txt files.",
        type=int,
        action='append',
        required=True,
    )
    parser.add_argument(
        "--kenlm_bins",
        help="File path to the KenLM binaries lmplz, filter and build_binary",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_order",
        help="Order of k-grams in ARPA-file generation",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_arpa_memory",
        help="Maximum allowed memory usage for ARPA-file generation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_prune",
        help="ARPA pruning parameters. Separate values with '|'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--binary_a_bits",
        help="Build binary quantization value a in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_q_bits",
        help="Build binary quantization value q in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_type",
        help="Build binary data structure type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--discount_fallback",
        help="To try when such message is returned by kenlm: 'Could not calculate Kneser-Ney "
             "discounts [...] rerun with --discount_fallback'",
        action="store_true",
    )

    args = parser.parse_args()

    # Basic checks on user supplied arguments
    if args.kenlm_bins == "path/to/kenlm/build/bin/":
        parser.error("Update kenlm_bins from documentation value to actual path for kenlm")

    if not os.path.isdir(args.kenlm_bins):
        parser.error("kenlm_bins must be a valid directory")

    kenlm_files = ["lmplz","filter","build_binary"]
    for f in kenlm_files:
        if not os.path.isfile(os.path.join(args.kenlm_bins, f)):
            parser.error(f"Required kenlm file {f} not found in kenlm_bins directory")

    if len(args.top_k) > len(args.input_txt):
        parser.error("Number of top_k arguments passed should not exceed number of input_txt"
                     " arguments passed")

    if not os.path.isdir(args.output_dir):
        parser.error("output_dir must be a valid directory")

    data_lower, vocab_str = convert_and_filter_topk(args)
    build_lm(args, data_lower, vocab_str)

    # Delete intermediate files
    os.remove(data_lower)
    os.remove(os.path.join(args.output_dir, "lm.arpa"))
    os.remove(os.path.join(args.output_dir, "lm_filtered.arpa"))


if __name__ == "__main__":
    main()
