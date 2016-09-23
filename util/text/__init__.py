import numpy as np
import tensorflow as tf

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space


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

    return tf.SparseTensor(indices=indices, values=values, shape=shape)
