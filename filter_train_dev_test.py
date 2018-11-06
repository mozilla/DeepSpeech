import pandas
import sys
import os


'''

USAGE:  $ python3 filter_cv1_dev_test.py LOCALE SAVE_TO_DIR
 e.g.:  $ python3 filter_cv1_dev_test.py 'ky' ../keep


The following script takes two files, clips.tsv and cv_LOCALE_valid.csv, and
figures out which clips have been validated and belong to each "bucket", aka
dev / test / train.

Then, after we know which validated clips belong to {valid / dev / train}, 
we save those three lists to three separate csv files.

For some reason, we have .mp3 files in the clips.tsv file, and .wav files in
the cv_LOCALE_valid.csv file.
'''


LOCALE = sys.argv[1]
output_folder = sys.argv[2]

ABS_PATH='/snakepit/shared/data/mozilla/CommonVoice/v2.0-alpha1.0'
#ABS_PATH='/home/josh/git/DeepSpeech/new'

# First, we import the main csv file which stores all the data for all languages,
# whether or not they've been validated (the file is called clips.tsv) 
# clips.tsv ==  path	sentence    up_votes	down_votes   age     gender	accent	locale	bucket
clips = pandas.read_csv('{}/clips.tsv'.format(ABS_PATH), sep='\t')
# pull out data for just one language
locale = clips[clips['locale'] == LOCALE]

# format file names and add abs path
locale['path'] = locale['path'].str.replace('/', '___')
locale['path'] = locale['path'].str.replace('mp3', 'wav')           # ugh
dev_paths = locale[locale['bucket'] == 'dev'].loc[:, ['path']]
test_paths = locale[locale['bucket'] == 'test'].loc[:, ['path']]
train_paths = locale[locale['bucket'] == 'train'].loc[:, ['path']]

dev_paths['path'] = '{}/{}/valid/'.format(ABS_PATH, LOCALE) + dev_paths['path'].astype(str)
test_paths['path'] = '{}/{}/valid/'.format(ABS_PATH, LOCALE) + test_paths['path'].astype(str)
train_paths['path'] = '{}/{}/valid/'.format(ABS_PATH, LOCALE) + train_paths['path'].astype(str)

# cv_LANG_valid.csv == wav_filename,wav_filesize,transcript
validated_clips = pandas.read_csv('{}/{}/cv_{}_valid.csv'.format(ABS_PATH, LOCALE, LOCALE))
validated_clips['wav_filename'] = '{}/{}/'.format(ABS_PATH, LOCALE) + validated_clips['wav_filename'].astype(str)
#validated_clips['transcript'] =  validated_clips['transcript'].str.replace(u'\xa0', ' ') # for ky only?

# produces a single column with a Bool for whether or not the validated clip is in dev / train / test
dev_indices = validated_clips['wav_filename'].isin(dev_paths['path'])
test_indices = validated_clips['wav_filename'].isin(test_paths['path'])
train_indices = validated_clips['wav_filename'].isin(train_paths['path'])

print("###############################################")
print("FILTERED CLIPS FOR THE LANGUAGE: ", str(LOCALE))
print("Num validated clips to be used in DEV: ", validated_clips[dev_indices]['wav_filename'].count())
print("Num validated clips to be used in TEST: ", validated_clips[test_indices]['wav_filename'].count())
print("Num validated clips to be used in TRAIN: ", validated_clips[train_indices]['wav_filename'].count())
print("###############################################")



validated_clips[dev_indices].to_csv(os.path.join(output_folder, 'cv_{}_valid_dev.csv'.format(LOCALE)), index=False)
validated_clips[test_indices].to_csv(os.path.join(output_folder, 'cv_{}_valid_test.csv'.format(LOCALE)), index=False)
validated_clips[train_indices].to_csv(os.path.join(output_folder, 'cv_{}_valid_train.csv'.format(LOCALE)), index=False)

