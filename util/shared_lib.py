from __future__ import print_function
from __future__ import absolute_import
from util.gpu import get_available_gpus
from ctypes import cdll
from sys import platform as _platform

def get_cupti_libname():
    if _platform == 'linux' or _platform == 'linux2':
        return 'libcupti.so'
    elif _platform == 'darwin':
        return 'libcupti.dylib'
    elif _platform == 'win32':
        return 'libcupti.dll'

def check_cupti():
    # We want to ensure that user has properly configured its libs.
    # We do this because dso load of libcupti will happen after a lot
    # of computation happened, so easy to miss and loose time.
    libname = get_cupti_libname()
    cupti = check_so(libname)
    if cupti is None:
        print("INFO: No %s because no GPU, go ahead." % libname)
    elif cupti is True:
        print("INFO: Found %s." % libname)
    else:
        print("WARNING: Running on GPU but no %s could be found ; will be unable to report GPU VRAM usage." % libname)

def check_so(soname):
    """
    Verify that we do have the 'soname' lib present in the system, and that it
    can be loaded.
    """

    if len(get_available_gpus()) == 0:
        return None

    # Try to force load lib, this would fail if the lib is not there :)
    try:
        lib = cdll.LoadLibrary(soname)
        print("INFO: Found so as", lib)
        assert lib.__class__.__name__ == 'CDLL'
        assert lib._name == soname
        return True
    except OSError as ex:
        print("WARNING:", ex)
        return False
