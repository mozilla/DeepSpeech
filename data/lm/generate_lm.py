import gzip
import io
import os
import subprocess
import tempfile

from collections import Counter
from urllib import request

def main():
  # Grab corpus.
  url = 'http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz'

  with tempfile.TemporaryDirectory() as tmp:
    data_upper = os.path.join(tmp, 'upper.txt.gz')
    print('Downloading {} into {}...'.format(url, data_upper))
    request.urlretrieve(url, data_upper)

    # Convert to lowercase and count word occurences.
    counter = Counter()
    data_lower = os.path.join(tmp, 'lower.txt.gz')
    print('Converting to lower case and counting word frequencies...')
    with io.TextIOWrapper(io.BufferedWriter(gzip.open(data_lower, 'w')), encoding='utf-8') as lower:
      with io.TextIOWrapper(io.BufferedReader(gzip.open(data_upper)), encoding='utf8') as upper:
        for line in upper:
          line_lower = line.lower()
          counter.update(line_lower.split())
          lower.write(line_lower)

    # Build pruned LM.
    lm_path = os.path.join(tmp, 'lm.arpa')
    print('Creating ARPA file...')
    subprocess.check_call([
      'lmplz', '--order', '5',
               '--temp_prefix', tmp,
               '--memory', '50%',
               '--text', data_lower,
               '--arpa', lm_path,
               '--prune', '0', '0', '1'
    ])

    # Filter LM using vocabulary of top 500k words
    filtered_path = os.path.join(tmp, 'lm_filtered.arpa')
    vocab_str = '\n'.join(word for word, count in counter.most_common(500000))
    print('Filtering ARPA file...')
    subprocess.run(['filter', 'single', 'model:{}'.format(lm_path), filtered_path], input=vocab_str.encode('utf-8'), check=True)

    # Quantize and produce trie binary.
    print('Building lm.binary...')
    subprocess.check_call([
      'build_binary', '-a', '255',
                      '-q', '8',
                      'trie',
                      filtered_path,
                      'lm.binary'
    ])

if __name__ == '__main__':
  main()
