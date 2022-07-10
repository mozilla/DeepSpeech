"""
A set of I/O utils that allow us to open files on remote storage as if they were present locally and access
into HDFS storage using Tensorflow's C++ FileStream API.
Currently only includes wrappers for Google's GCS, but this can easily be expanded for AWS S3 buckets.
"""
import os
from tensorflow.io import gfile


def is_remote_path(path):
    """
    Returns True iff the path is one of the remote formats that this
    module supports
    """
    return path.startswith('gs://') or path.startswith('hdfs://')


def path_exists_remote(path):
    """
    Wrapper that allows existance check of local and remote paths like
    `gs://...`
    """
    if is_remote_path(path):
        return gfile.exists(path)
    return os.path.exists(path)


def copy_remote(src, dst, overwrite=False):
    """
    Allows us to copy a file from local to remote or vice versa
    """
    return gfile.copy(src, dst, overwrite)


def open_remote(path, mode='r', buffering=-1, encoding=None, newline=None, closefd=True, opener=None):
    """
    Wrapper around open() method that can handle remote paths like `gs://...`
    off Google Cloud using Tensorflow's IO helpers.

    buffering, encoding, newline, closefd, and opener are ignored for remote files

    This enables us to do:
    with open_remote('gs://.....', mode='w+') as f:
        do something with the file f, whether or not we have local access to it
    """
    if is_remote_path(path):
        return gfile.GFile(path, mode=mode)
    return open(path, mode, buffering=buffering, encoding=encoding, newline=newline, closefd=closefd, opener=opener)


def isdir_remote(path):
    """
    Wrapper to check if remote and local paths are directories
    """
    if is_remote_path(path):
        return gfile.isdir(path)
    return os.path.isdir(path)


def listdir_remote(path):
    """
    Wrapper to list paths in local dirs (alternative to using a glob, I suppose)
    """
    if is_remote_path(path):
        return gfile.listdir(path)
    return os.listdir(path)


def glob_remote(filename):
    """
    Wrapper that provides globs on local and remote paths like `gs://...`
    """
    return gfile.glob(filename)


def remove_remote(filename):
    """
    Wrapper that can remove local and remote files like `gs://...`
    """
    # Conditional import
    return gfile.remove(filename)
