import os
import platform
import sys
from pathlib import Path

from pkg_resources import parse_version
from setuptools import find_packages, setup


def get_decoder_pkg_url(version, artifacts_root=None):
    is_arm = 'arm' in platform.machine()
    is_mac = 'darwin' in sys.platform
    is_64bit = sys.maxsize > (2**31 - 1)

    if is_arm:
        tc_arch = 'arm64-ctc' if is_64bit else 'arm-ctc'
    elif is_mac:
        tc_arch = 'osx-ctc'
    else:
        tc_arch = 'cpu-ctc'

    ds_version = parse_version(version)
    branch = "v{}".format(version)

    plat = platform.system().lower()
    arch = platform.machine()

    if plat == 'linux' and arch == 'x86_64':
        plat = 'manylinux1'

    if plat == 'darwin':
        plat = 'macosx_10_10'

    is_ucs2 = sys.maxunicode < 0x10ffff
    m_or_mu = 'mu' if is_ucs2 else 'm'

    # ABI does not contain m / mu anymore after Python 3.8
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        m_or_mu = ''

    pyver = ''.join(str(i) for i in sys.version_info[0:2])

    if not artifacts_root:
        artifacts_root = 'https://community-tc.services.mozilla.com/api/index/v1/task/project.deepspeech.deepspeech.native_client.{branch_name}.{tc_arch_string}/artifacts/public'.format(
            branch_name=branch,
            tc_arch_string=tc_arch)

    return 'ds_ctcdecoder @ {artifacts_root}/ds_ctcdecoder-{ds_version}-cp{pyver}-cp{pyver}{m_or_mu}-{platform}_{arch}.whl'.format(
        artifacts_root=artifacts_root,
        ds_version=ds_version,
        pyver=pyver,
        m_or_mu=m_or_mu,
        platform=plat,
        arch=arch,
    )


def main():
    version_file = Path(__file__).parent / 'VERSION'
    with open(str(version_file)) as fin:
        version = fin.read().strip()

    decoder_pkg_url = get_decoder_pkg_url(version)

    install_requires_base = [
        'tensorflow == 1.15.2',
        'numpy',
        'progressbar2',
        'six',
        'pyxdg',
        'attrdict',
        'absl-py',
        'semver',
        'opuslib == 2.0.0',
        'optuna',
        'sox',
        'bs4',
        'pandas',
        'requests',
        'numba == 0.47.0', # ships py3.5 wheel
        'llvmlite == 0.31.0', # for numba==0.47.0
        'librosa',
        'soundfile',
    ]

    # Due to pip craziness environment variables are the only consistent way to
    # get options into this script when doing `pip install`.
    tc_decoder_artifacts_root = os.environ.get('DECODER_ARTIFACTS_ROOT', '')
    if tc_decoder_artifacts_root:
        # We're running inside the TaskCluster environment, override the decoder
        # package URL with the one we just built.
        decoder_pkg_url = get_decoder_pkg_url(version, tc_decoder_artifacts_root)
        install_requires = install_requires_base + [decoder_pkg_url]
    elif os.environ.get('DS_NODECODER', ''):
        install_requires = install_requires_base
    else:
        install_requires = install_requires_base + [decoder_pkg_url]

    setup(
        name='deepspeech_training',
        version=version,
        description='Training code for mozilla DeepSpeech',
        url='https://github.com/mozilla/DeepSpeech',
        author='Mozilla',
        license='MPL-2.0',
        # Classifiers help users find your project by categorizing it.
        #
        # For a list of valid classifiers, see https://pypi.org/classifiers/
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Multimedia :: Sound/Audio :: Speech',
            'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
            'Programming Language :: Python :: 3',
        ],
        package_dir={'': 'training'},
        packages=find_packages(where='training'),
        python_requires='>=3.5, <4',
        install_requires=install_requires,
        # If there are data files included in your packages that need to be
        # installed, specify them here.
        package_data={
            'deepspeech_training': [
                'VERSION',
                'GRAPH_VERSION',
            ],
        },
    )

if __name__ == '__main__':
    main()
