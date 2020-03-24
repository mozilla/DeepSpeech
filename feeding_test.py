# Add util/ as loadable
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import absl
import tensorflow.compat.v1 as tfv1
tfv1.enable_eager_execution()

from util.config import initialize_globals
from util.flags import create_flags
from util.feeding import create_dataset

def test_load(name, amount, limit=0):
    dataset = create_dataset(['util/test_data/sample_collections_2.csv', 'util/test_data/sample_collections_2.sdb'],
                             batch_size=1,
                             limit=limit,
                             enable_cache=False)
    elems = len(list(dataset))
    print('Asserting [{}] {} == {} (limit={}) ...'.format(name, amount, elems, limit), end=' ')
    assert amount == elems
    print('OK')

def run_tests(_):
    initialize_globals()

    test_load('test_no_limit', 4)
    test_load('test_limit_None', 4, limit=None)
    test_load('test_limit_inf', 3, limit=3)
    test_load('test_limit_inf2', 1, limit=1)
    test_load('test_limit_sup', 4, limit=8)
    test_load('test_limit_neg', 1, limit=-1)
    test_load('test_limit_neg2', 4, limit=-8)

if __name__ == '__main__':
    create_flags()
    absl.app.run(run_tests)
