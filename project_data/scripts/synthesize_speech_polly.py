"""
This script will generate the speech from AWS Polly service
"""

from config import *
from utils import *
import os
import sys
import boto3


def synthesize_from_polly(keyword, polly_client, filename_prefix):
    mp3_file = '%s.mp3' % filename_prefix
    wav_file = '%s.wav' % filename_prefix
    response_ok = True
    with open(mp3_file, 'wb') as mp3file:
        response = polly_client.synthesize_speech(VoiceId='Joey',
                                                  OutputFormat='mp3',
                                                  SampleRate='16000',
                                                  Text=keyword)
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            mp3file.write(response['AudioStream'].read())
            print('Handled %s' % mp3_file)
            response_ok = True
    if response_ok:
        cmd = "ffmpeg -f mp3 -i %s -ar 16000 -ac 1 -y %s" % (mp3_file, wav_file)
        # print(cmd)
        os.system(cmd)
        print("Converted: %s" % wav_file)


def load_phrases_dict(n_gram):
    json_filename = "%s_%d.json" % (Test_Keywords_Filename_Prefix, n_gram)
    return load_json(json_filename)


def prepare_test_audio(phrase_dict, n_gram, polly_client):
    target_directory = os.path.join(Test_Corpora_Home, "%d_gram" % n_gram)
    for key in phrase_dict:
        filename_prefix = os.path.join(target_directory, key)
        synthesize_from_polly(phrase_dict[key], polly_client, filename_prefix)


if __name__ == '__main__':
    assert len(sys.argv) == 3
    public_key = sys.argv[1]
    private_key = sys.argv[2]
    aws_polly_client = boto3.Session(
        aws_access_key_id=public_key,
        aws_secret_access_key=private_key,
        region_name='us-west-2').client('polly')

    n_gram_list = [1, 6]
    for ng in n_gram_list:
        phrases_to_be_said = load_phrases_dict(ng)
        prepare_test_audio(phrases_to_be_said, ng, aws_polly_client)
