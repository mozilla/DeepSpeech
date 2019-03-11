import pandas

file1 = pandas.read_csv('test.csv', encoding='utf-8', na_filter=False)
print(file1.keys())
print(file1)
#print(file1.tran)
print(file1[['transcript', 'wav_filename']])
#print(file1['wav_filename'])
