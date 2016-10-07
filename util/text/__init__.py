import numpy as np
import tensorflow as tf

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

def text_to_sparse_tensor(originals):
    return tf.SparseTensor.from_value(text_to_sparse_tensor_value(originals))

def text_to_sparse_tensor_value(originals):
    tuple = text_to_sparse_tuple(originals)
    return tf.SparseTensorValue(indices=tuple[0], values=tuple[1], shape=tuple[2])

def text_to_sparse_tuple(originals):
    # Define list to hold results
    results = []

    # Process each original in originals
    for original in originals:
        # Create list of sentence's words w/spaces replaced by ''
        result = original.replace(' ', '  ')
        result = result.split(' ')

        # Tokenize words into letters adding in SPACE_TOKEN where required
        result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])

        # Map characters into indicies
        result = np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])

        # Add result to results
        results.append(result)

    # Creating sparse representation to feed the placeholder
    return sparse_tuple_from(results)

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), xrange(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return (indices, values, shape);

def sparse_tensor_value_to_text(value):
    return sparse_tuple_to_text((value.indices, value.values, value.shape))

def sparse_tuple_to_text(tuple):
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        c = ' ' if c == SPACE_INDEX else chr(c + FIRST_INDEX)
        results[index] = results[index] + c
    # List of strings
    return results

def wer(original, result):
    return levenshtein(original, result) / float(len(original.split(' ')))

def wers(originals, results):
    count = len(originals)
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = wer(originals[i], results[i])
        mean = mean + rate
        rates.append(mean)
    return rates, mean / float(count)

# The following code is from: http://hetland.org/coding/python/levenshtein.py

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
