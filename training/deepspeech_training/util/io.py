"""
A set of I/O utils that allow us to open files on remote storage as if they were present locally.
Currently only includes wrappers for Google's GCS, but this can easily be expanded for AWS S3 buckets.
"""
import inspect
import os
import sys

def path_exists_remote(path):
    """
    Wrapper that allows existance check of local and remote paths like
    `gs://...`
    """
    # Conditional import
    if path.startswith("gs://"):
        from tensorflow.io import gfile
        return gfile.exists(path)
    return os.path.exists(path)


def open_remote(path, mode):
    """
    Wrapper around open_remote() method that can handle remote paths like `gs://...`
    off Google Cloud using Tensorflow's IO helpers.

    This enables us to do:
    with open_remote('gs://.....', mode='w+') as f:
        do something with the file f, whether or not we have local access to it
    """
    # Conditional import
    if path.startswith("gs://"):
        from tensorflow.io import gfile
        return gfile.GFile(path, mode=mode)
    return open_remote(path, mode)


def isdir_remote(path):
    """
    Wrapper to check if remote and local paths are directories
    """
    # Conditional import
    if path.startswith("gs://"):
        from tensorflow.io import gfile
        return gfile.isdir(path)
    return os.path.isdir(path)


def listdir_remote(path):
    """
    Wrapper to list paths in local dirs (alternative to using a glob, I suppose)
    """
    # Conditional import
    if path.startswith("gs://"):
        from tensorflow.io import gfile
        return gfile.listdir(path)
    return os.listdir(path)


def glob_remote(filename):
    """
    Wrapper that provides globs on local and remote paths like `gs://...`
    """
    # Conditional import
    from tensorflow.io import gfile

    return gfile.glob(filename)


def remove_remote(filename):
    """
    Wrapper that can remove_remote local and remote files like `gs://...`
    """
    # Conditional import
    from tensorflow.io import gfile

    return gfile.remove_remote(filename)