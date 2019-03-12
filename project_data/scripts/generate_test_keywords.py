"""
This script is responsible for generating n_gram keywords with respect to the speaker_id
"""

from config import *
from utils import *
import enchant
import os
from collections import defaultdict
import random

En_US_Dict = enchant.Dict("en_US")
All_Phrases_List_Filename_prefix = "all_phrases"


def check_in_eng_dict(phrase):
    """
    This method will test if all the words in the phrase is present in the en-US dictionary
    :param phrase: phrase which will be checked for correctness
    :return: true if all the words are present in the dictionary false otherwise
    """
    return ' '.join([wrd for wrd in phrase.split() if En_US_Dict.check(wrd)]) == phrase


def extract_n_grams(phrase, n_gram):
    all_words = phrase.split()
    all_phrases = [all_words[x:x + n_gram] for x in range(len(all_words))]
    return [' '.join(x) for x in all_phrases if len(x) == n_gram]


def get_all_phrases_by_fileid(n_gram):
    result = defaultdict(list)
    for text_filename in list_all_files_with_ext(Librispeech_Home, Text_Ext):
        with open(os.path.join(Librispeech_Home, text_filename)) as tf:
            fileid = text_filename.split(Text_Ext)[0]
            for line in tf.read().splitlines():
                result[fileid].extend(extract_n_grams(line, n_gram))
    return result


def generate_phrase_prefix(true_pos, in_eng_dict):
    if true_pos and in_eng_dict:
        return "TP_ID_"
    elif not true_pos and in_eng_dict:
        return "TN_ID_"
    elif true_pos and not in_eng_dict:
        return "TP_OD_"
    else:
        return "TN_OD_"


def generate_phrases(num_phrases, n_gram, true_pos, in_eng_dict):
    all_phrases = load_json("%s_%d.json" % (All_Phrases_List_Filename_prefix, n_gram))
    true_pos_fileids = set(get_ids_for_speaker(Speaker_Id))
    true_pos_phrases = []
    true_neg_phrases = []
    for fileid in all_phrases:
        if fileid in true_pos_fileids:
            true_pos_phrases.extend(all_phrases[fileid])
        else:
            true_neg_phrases.extend(all_phrases[fileid])
    # print("Generated %d true_positive phrases" % len(true_pos_phrases))     # 812 for 6-gram in our case
    # print("Generated %d true_negative phrases" % len(true_neg_phrases))     # 40274 for 6-gram in our case

    sample_set = true_pos_phrases if true_pos else true_neg_phrases
    random.shuffle(sample_set)

    if in_eng_dict:
        filtered = [phrase for phrase in sample_set if check_in_eng_dict(phrase)]
    else:
        filtered = [phrase for phrase in sample_set if not check_in_eng_dict(phrase)]

    return list(set(filtered))[:num_phrases]


if __name__ == '__main__':
    one_gram_all_phrases = get_all_phrases_by_fileid(1)
    six_gram_all_phrases = get_all_phrases_by_fileid(6)
    dump_json(one_gram_all_phrases, "%s_1.json" % All_Phrases_List_Filename_prefix)
    dump_json(six_gram_all_phrases, "%s_6.json" % All_Phrases_List_Filename_prefix)

    test_configs = [(1, True, True), (1, True, False), (1, False, True), (1, False, False),
                    (6, True, True), (6, True, False), (6, False, True), (6, False, False)]

    num_test_phrases = 100
    overall_dict = defaultdict(dict)
    for ng, tp, id in test_configs:
        test_phrases = generate_phrases(num_test_phrases, ng, tp, id)
        print("Generating for n-gram: %d, true_pos: %s, in_eng_dict: %s with num_samples: %d" %
              (ng, str(tp), str(id), len(test_phrases)))
        phrase_prefix = generate_phrase_prefix(tp, id)
        phrase_dict = {"%s%d" % (phrase_prefix, idx): val for idx, val in enumerate(test_phrases)}
        overall_dict[ng].update(phrase_dict)

    for ng in overall_dict:
        complete_filename = "%s_%d.json" % (Test_Keywords_Filename_Prefix, ng)
        dump_json(overall_dict[ng], complete_filename)


