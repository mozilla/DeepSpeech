
def keep_only_digits(txt):
    return ''.join(filter(lambda c: c.isdigit(), txt))


def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)

# pylint: disable=import-outside-toplevel
def check_ctcdecoder_version():
    import sys
    import os
    import semver

    ds_version_s = open(os.path.join(os.path.dirname(__file__), '../VERSION')).read().strip()

    try:
        from ds_ctcdecoder import __version__ as decoder_version
    except ImportError as e:
        if e.msg.find('__version__') > 0:
            print("DeepSpeech version ({ds_version}) requires CTC decoder to expose __version__. Please upgrade the ds_ctcdecoder package to version {ds_version}".format(ds_version=ds_version_s))
            sys.exit(1)
        raise e

    decoder_version_s = decoder_version.decode()

    rv = semver.compare(ds_version_s, decoder_version_s)
    if rv != 0:
        print("DeepSpeech version ({}) and CTC decoder version ({}) do not match. Please ensure matching versions are in use.".format(ds_version_s, decoder_version_s))
        sys.exit(1)

    return rv
