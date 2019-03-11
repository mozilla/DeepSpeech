"""
This file will contain the global configs necessary for the scripts
"""

Global_Root = "/home/ubuntu/data"
Librispeech_Home = "%s/orig_audio" % Global_Root
Json_Home = "%s/intermediate_results" % Global_Root
Audio_Corpora_Home = "%s/final_corpora" % Global_Root
Test_Corpora_Home = "%s/final_test_data" % Global_Root

Speaker_Ext = ".spk"
Flac_Ext = ".flac"
Text_Ext = ".wrd"

Speaker_Id = "3752"     # This speaker has got the highest number of audio files (110)
Test_Keywords_Filename_Prefix = "test_phrases"

