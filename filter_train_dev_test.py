import pandas
import sys
import os
import ntpath

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



## Expected to exist:
#
# input_folder/
#     LOCALE/
#         valid/
#             *.wav
#         cv_LOCALE_valid.csv (the wav_filename actually includes "valid/")
#     clips.tsv

## Created by script:
#
# output_folder/
#     cv_valid_dev.csv
#     cv_valid_test.csv
#     cv_valid_train.csv




LOCALE = sys.argv[1]
input_folder =  sys.argv[2]
output_folder = sys.argv[3]



####           ####
#### CLIPS.TSV ####
####           ####


# First, we import the main csv file which stores all the data for all languages,
# whether or not they've been validated (the file is called clips.tsv) 
# clips.tsv ==  path	sentence    up_votes	down_votes   age     gender	accent	locale	bucket

print("Looking for clips.tsv here: ", '{}/clips.tsv'.format(input_folder))
clips = pandas.read_csv('{}/clips.tsv'.format(input_folder), sep='\t')
# pull out data for just one language
locale = clips[clips['locale'] == LOCALE]
# format file names
locale['path'] = locale['path'].str.replace('/', '___')
locale['path'] = locale['path'].str.replace('mp3', 'wav')
dev_paths = locale[locale['bucket'] == 'dev'].loc[:, ['path']]
test_paths = locale[locale['bucket'] == 'test'].loc[:, ['path']]
train_paths = locale[locale['bucket'] == 'train'].loc[:, ['path']]



####                    ####
#### CV_LOCAL_VALID.TSV ####
####                    ####

# cv_LANG_valid.csv == wav_filename,wav_filesize,transcript

validated_clips = pandas.read_csv('{}/{}/cv_{}_valid.csv'.format(input_folder, LOCALE, LOCALE))
validated_clips['path'] = validated_clips['wav_filename'].apply(ntpath.basename)
validated_clips['transcript'] =  validated_clips['transcript'].str.replace(u'\xa0', ' ') # for ky only?




####              ####
#### EXTRACT SETS ####
####              ####

# produces a single column with a Bool for whether or not the validated clip is in dev / train / test
dev_indices = validated_clips['path'].isin(dev_paths['path'])
test_indices = validated_clips['path'].isin(test_paths['path'])
train_indices = validated_clips['path'].isin(train_paths['path'])
validated_clips = validated_clips.drop(columns=['path'])

print("###############################################")
print("FILTERED CLIPS FOR THE LANGUAGE: ", str(LOCALE))
print("Num validated clips to be used in DEV: ", validated_clips[dev_indices]['wav_filename'].count())
print("Num validated clips to be used in TEST: ", validated_clips[test_indices]['wav_filename'].count())
print("Num validated clips to be used in TRAIN: ", validated_clips[train_indices]['wav_filename'].count())
print("###############################################")

ABS_PATH_TO_DATA = input_folder + "/" + LOCALE + "/"
validated_clips['wav_filename'] =  ABS_PATH_TO_DATA + validated_clips['wav_filename'].astype(str)

validated_clips[dev_indices].to_csv(os.path.join(output_folder, 'valid_dev.csv'), index=False)
validated_clips[test_indices].to_csv(os.path.join(output_folder, 'valid_test.csv'), index=False)
validated_clips[train_indices].to_csv(os.path.join(output_folder, 'valid_train.csv'), index=False)

