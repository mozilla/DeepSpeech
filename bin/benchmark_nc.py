#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import sys

# To use util.tc
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(sys.argv[0]))))
import util.taskcluster as tcu
from util.benchmark import keep_only_digits

import paramiko
import argparse
import tempfile
import shutil
import subprocess
import stat
import numpy
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
import csv
import getpass
import zipfile

from six import iteritems
from six.moves import range, map

r'''
 Tool to:
  - remote local or remote (ssh) native_client
  - handles copying models (as protocolbuffer files)
  - run native_client in benchmark mode
  - collect timing results
  - compute mean values (with wariances)
  - output as CSV
'''

ssh_conn = None
def exec_command(command, cwd=None):
    r'''
    Helper to exec locally (subprocess) or remotely (paramiko)
    '''

    rc = None
    stdout = stderr = None
    if ssh_conn is None:
        ld_library_path = {'LD_LIBRARY_PATH': '.:%s' % os.environ.get('LD_LIBRARY_PATH', '')}
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=ld_library_path, cwd=cwd)
        stdout, stderr = p.communicate()
        rc = p.returncode
    else:
        # environment= requires paramiko >= 2.1 (fails with 2.0.2)
        final_command = command if cwd is None else 'cd %s && %s %s' % (cwd, 'LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH', command)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh_conn.exec_command(final_command)
        stdout = ''.join(ssh_stdout.readlines())
        stderr = ''.join(ssh_stderr.readlines())
        rc = ssh_stdout.channel.recv_exit_status()

    return rc, stdout, stderr

def assert_valid_dir(dir):
    if dir is None:
        raise AssertionError('Invalid temp directory')
    return True

def get_arch_string():
    r'''
    Check local or remote system arch, to produce TaskCluster proper link.
    '''
    rc, stdout, stderr = exec_command('uname -sm')
    if rc > 0:
        raise AssertionError('Error checking OS')

    stdout = stdout.lower().strip()
    if not 'linux' in stdout:
        raise AssertionError('Unsupported OS')

    if 'armv7l' in stdout:
        return 'arm'

    if 'x86_64' in stdout:
        nv_rc, nv_stdout, nv_stderr = exec_command('nvidia-smi')
        nv_stdout = nv_stdout.lower().strip()
        if 'NVIDIA-SMI' in nv_stdout:
            return 'gpu'
        else:
            return 'cpu'

    raise AssertionError('Unsupported arch:', stdout)

def maybe_download_binaries(dir):
    assert_valid_dir(dir)
    tcu.maybe_download_tc(target_dir=dir, tc_url=tcu.get_tc_url(get_arch_string()), progress=True)

def extract_native_client_tarball(dir):
    r'''
    Download a native_client.tar.xz file from TaskCluster and extract it to dir.
    '''
    assert_valid_dir(dir)

    target_tarball = os.path.join(dir, 'native_client.tar.xz')
    if os.path.isfile(target_tarball) and os.stat(target_tarball).st_size == 0:
        return

    subprocess.check_call(['pixz', '-d', 'native_client.tar.xz'], cwd=dir)
    subprocess.check_call(['tar', 'xf', 'native_client.tar'], cwd=dir)
    os.unlink(os.path.join(dir, 'native_client.tar'))
    open(target_tarball, 'w').close()

def is_zip_file(models):
    r'''
    Ensure that a path is a zip file by:
     - checking length is 1
     - checking extension is '.zip'
    '''
    ext = os.path.splitext(models[0])[1]
    return (len(models) == 1) and (ext == '.zip')

def maybe_inspect_zip(models):
    r'''
    Detect if models is a list of protocolbuffer files or a ZIP file.
    If the latter, then unzip it and return the list of protocolbuffer files
    that were inside.
    '''

    if len(models) > 1:
        return models

    # With AOT, we may have just one file that is not a ZIP file
    # so verify that we don't have a .zip extension
    if not(is_zip_file(models)):
        return models

    if len(models) < 1:
        raise AssertionError('No models at all')

    return zipfile.ZipFile(models[0]).namelist()

def all_files(models=[]):
    r'''
    Return a list of full path of files matching 'models', sorted in human
    numerical order (i.e., 0 1 2 ..., 10 11 12, ..., 100, ..., 1000).

    Files are supposed to be named identically except one variable component
    e.g. the list,
      test.weights.e5.lstm1200.ldc93s1.pb
      test.weights.e5.lstm1000.ldc93s1.pb
      test.weights.e5.lstm800.ldc93s1.pb
    gets sorted:
      test.weights.e5.lstm800.ldc93s1.pb
      test.weights.e5.lstm1000.ldc93s1.pb
      test.weights.e5.lstm1200.ldc93s1.pb
    '''

    def nsort(a, b):
        fa = os.path.basename(a).split('.')
        fb = os.path.basename(b).split('.')
        elements_to_remove = []

        assert len(fa) == len(fb)

        for i in range(0, len(fa)):
            if fa[i] == fb[i]:
                elements_to_remove.append(fa[i])

        for e in elements_to_remove:
            fa.remove(e)
            fb.remove(e)

        assert len(fa) == len(fb)
        assert len(fa) == 1

        fa = keep_only_digits(fa[0])
        fb = keep_only_digits(fb[0])

        if fa < fb:
            return -1
        if fa == fb:
            return 0
        if fa > fb:
            return 1

    base = list(map(lambda x: os.path.abspath(x), maybe_inspect_zip(models)))
    base.sort(cmp=nsort)

    return base

def copy_tree(dir):
    assert_valid_dir(dir)

    sftp = ssh_conn.open_sftp()
    # IOError will get triggered if the path does not exists remotely
    try:
        if stat.S_ISDIR(sftp.stat(dir).st_mode):
            print('Directory already existent: %s' % dir)
    except IOError:
        print('Creating remote directory: %s' % dir)
        sftp.mkdir(dir)

    print('Copy files to remote')
    for fname in os.listdir(dir):
        fullpath = os.path.join(dir, fname)
        local_stat  = os.stat(fullpath)
        try:
            remote_mode = sftp.stat(fullpath).st_mode
        except IOError:
            remote_mode = 0

        if not stat.S_ISREG(remote_mode):
            print('Copying %s ...' % fullpath)
            remote_mode = sftp.put(fullpath, fullpath, confirm=True).st_mode

        if local_stat.st_mode != remote_mode:
            print('Setting proper remote mode: %s' % local_stat.st_mode)
            sftp.chmod(fullpath, local_stat.st_mode)

    sftp.close()

def delete_tree(dir):
    assert_valid_dir(dir)

    sftp = ssh_conn.open_sftp()
    # IOError will get triggered if the path does not exists remotely
    try:
        if stat.S_ISDIR(sftp.stat(dir).st_mode):
            print('Removing remote files')
            for fname in sftp.listdir(dir):
                fullpath = os.path.join(dir, fname)
                remote_stat = sftp.stat(fullpath)
                if stat.S_ISREG(remote_stat.st_mode):
                    print('Removing %s ...' % fullpath)
                    sftp.remove(fullpath)

            print('Removing directory %s ...' % dir)
            sftp.rmdir(dir)

        sftp.close()
    except IOError:
        print('No remote directory: %s' % dir)

def setup_tempdir(dir, models, wav, alphabet, lm_binary, trie, binaries):
    r'''
    Copy models, libs and binary to a directory (new one if dir is None)
    '''
    if dir is None:
        dir = tempfile.mkdtemp(suffix='dsbench')

    sorted_models = all_files(models=models)
    if binaries is None:
        maybe_download_binaries(dir)
    else:
        print('Using local binaries: %s' % (binaries))
        shutil.copy2(binaries, dir)
    extract_native_client_tarball(dir)

    filenames = map(lambda x: os.path.join(dir, os.path.basename(x)), sorted_models)
    missing_models = filter(lambda x: not os.path.isfile(x), filenames)
    if len(missing_models) > 0:
        # If we have a ZIP file, directly extract it to the proper path
        if is_zip_file(models):
            print('Extracting %s to %s' % (models[0], dir))
            zipfile.ZipFile(models[0]).extractall(path=dir)
            print('Extracted %s.' % models[0])
        else:
            # If one model is missing, let's copy everything again. Be safe.
            for f in sorted_models:
                print('Copying %s to %s' % (f, dir))
                shutil.copy2(f, dir)

    for extra_file in [ wav, alphabet, lm_binary, trie ]:
        if extra_file and not os.path.isfile(os.path.join(dir, os.path.basename(extra_file))):
            print('Copying %s to %s' % (extra_file, dir))
            shutil.copy2(extra_file, dir)

    if ssh_conn:
        copy_tree(dir)

    return dir, sorted_models

def teardown_tempdir(dir):
    r'''
    Cleanup temporary directory.
    '''

    if ssh_conn:
        delete_tree(dir)

    assert_valid_dir(dir)
    shutil.rmtree(dir)

def get_sshconfig():
    r'''
    Read user's SSH configuration file
    '''

    with open(os.path.expanduser('~/.ssh/config')) as f:
        cfg = paramiko.SSHConfig()
        cfg.parse(f)
        ret_dict = {}
        for d in cfg._config:
            _copy = dict(d)
            # Avoid buggy behavior with strange host definitions, we need
            # Hostname and not Host.
            del _copy['host']
            for host in d['host']:
                ret_dict[host] = _copy['config']

        return ret_dict

def establish_ssh(target=None, auto_trust=False, allow_agent=True, look_keys=True):
    r'''
    Establish a SSH connection to a remote host. It should be able to use
    SSH's config file Host name declarations. By default, will not automatically
    add trust for hosts, will use SSH agent and will try to load keys.
    '''

    def password_prompt(username, hostname):
        r'''
        If the Host is relying on password authentication, lets ask it.
        Relying on SSH itself to take care of that would not work when the
        remote authentication is password behind a SSH-key+2FA jumphost.
        '''
        return getpass.getpass('No SSH key for %s@%s, please provide password: ' % (username, hostname))

    ssh_conn = None
    if target is not None:
        ssh_conf = get_sshconfig()
        cfg = {
            'hostname': None,
            'port': 22,
            'allow_agent': allow_agent,
            'look_for_keys': look_keys
        }
        if ssh_conf.has_key(target):
            user_config = ssh_conf.get(target)

            # If ssh_config file's Host defined 'User' instead of 'Username'
            if user_config.has_key('user') and not user_config.has_key('username'):
                user_config['username'] = user_config['user']
                del user_config['user']

            for k in ('username', 'hostname', 'port'):
                if k in user_config:
                    cfg[k] = user_config[k]

            # Assume Password auth. If we don't do that, then when connecting
            # through a jumphost we will run into issues and the user will
            # not be able to input his password to the SSH prompt.
            if 'identityfile' in user_config:
                cfg['key_filename'] = user_config['identityfile']
            else:
                cfg['password'] = password_prompt(cfg['username'], cfg['hostname'] or target)

            # Should be the last one, since ProxyCommand will issue connection to remote host
            if 'proxycommand' in user_config:
                cfg['sock'] = paramiko.ProxyCommand(user_config['proxycommand'])

        else:
            cfg['username'] = target.split('@')[0]
            cfg['hostname'] = target.split('@')[1].split(':')[0]
            cfg['password'] = password_prompt(cfg['username'], cfg['hostname'])
            try:
                cfg['port'] = int(target.split('@')[1].split(':')[1])
            except IndexError:
                # IndexError will happen if no :PORT is there.
                # Default value 22 is defined above in 'cfg'.
                pass

        ssh_conn = paramiko.SSHClient()
        if auto_trust:
            ssh_conn.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh_conn.connect(**cfg)

    return ssh_conn

def run_benchmarks(dir, models, wav, alphabet, lm_binary=None, trie=None, iters=-1, extra_aot_model=None):
    r'''
    Core of the running of the benchmarks. We will run on all of models, against
    the WAV file provided as wav, and the provided alphabet.

    If supplied extra_aot_model, add another pass with the .so built AOT model.
    '''

    assert_valid_dir(dir)

    inference_times = [ ]

    if extra_aot_model:
        models.append(extra_aot_model)

    for model in models:
        model_filename = '' if model is extra_aot_model else model

        current_model = {
          'name':   model,
          'iters':  [ ],
          'mean':   numpy.infty,
          'stddev': numpy.infty
        }

        if lm_binary and trie:
            cmdline = './deepspeech --model "%s" --alphabet "%s" --lm "%s" --trie "%s" --audio "%s" -t' % (model_filename, alphabet, lm_binary, trie, wav)
        else:
            cmdline = './deepspeech --model "%s" --alphabet "%s" --audio "%s" -t' % (model_filename, alphabet, wav)

        for it in range(iters):
            sys.stdout.write('\rRunning %s: %d/%d' % (os.path.basename(model), (it+1), iters))
            sys.stdout.flush()
            rc, stdout, stderr = exec_command(cmdline, cwd=dir)
            if rc == 0:
                inference_time = float(stdout.split('\n')[1].split('=')[-1])
                # print("[%d] model=%s inference=%f" % (it, model, inference_time))
                current_model['iters'].append(inference_time)
            else:
                print('exec_command("%s") failed with rc=%d' % (cmdline, rc))
                print('stdout: %s' % stdout)
                print('stderr: %s' % stderr)
                raise AssertionError('Execution failure: rc=%d' % (rc))

        sys.stdout.write('\n')
        sys.stdout.flush()
        current_model['mean']   = numpy.mean(current_model['iters'])
        current_model['stddev'] = numpy.std(current_model['iters'])
        inference_times.append(current_model)

    return inference_times

def produce_csv(input, output):
    r'''
    Take an input dictionnary and write it to the object-file output.
    '''
    output.write('"model","mean","std"\n')
    for model_data in input:
        output.write('"%s",%f,%f\n' % (model_data['name'], model_data['mean'], model_data['stddev']))
    output.flush()
    output.close()
    print("Wrote as %s" % output.name)

def handle_args():
    parser = argparse.ArgumentParser(description='Benchmarking tooling for DeepSpeech native_client.')
    parser.add_argument('--target', required=False,
                                 help='SSH user:pass@host string for remote benchmarking. This can also be a name of a matching \'Host\' in your SSH config.')
    parser.add_argument('--autotrust', action='store_true', default=False,
                                 help='SSH Paramiko policy to automatically trust unknown keys.')
    parser.add_argument('--allowagent', action='store_true', dest='allowagent',
                                 help='Allow the use of a SSH agent.')
    parser.add_argument('--no-allowagent', action='store_false', dest='allowagent',
                                 help='Disallow the use of a SSH agent.')
    parser.add_argument('--lookforkeys', action='store_true', dest='lookforkeys',
                                 help='Allow to look for SSH keys in ~/.ssh/.')
    parser.add_argument('--no-lookforkeys', action='store_false', dest='lookforkeys',
                                 help='Disallow to look for SSH keys in ~/.ssh/.')
    parser.add_argument('--dir', required=False, default=None,
                                 help='Local directory where to copy stuff. This will be mirrored to the remote system if needed (make sure to use path that will work on both).')
    parser.add_argument('--models', nargs='+', required=False,
                                 help='List of files (protocolbuffer) to work on. Might be a zip file.')
    parser.add_argument('--so-model', required=False,
                                 help='Perform one step using AOT-based .so model')
    parser.add_argument('--wav', required=False,
                                 help='WAV file to pass to native_client. Supply again in plotting mode to draw realine line.')
    parser.add_argument('--alphabet', required=False,
                                 help='Text file to pass to native_client for the alphabet.')
    parser.add_argument('--lm_binary', required=False,
                                 help='Path to the LM binary file used by the decoder.')
    parser.add_argument('--trie', required=False,
                                 help='Path to the trie file used by the decoder.')
    parser.add_argument('--iters', type=int, required=False, default=5,
                                 help='How many iterations to perfom on each model.')
    parser.add_argument('--keep', required=False, action='store_true',
                                 help='Keeping run files (binaries & models).')
    parser.add_argument('--csv', type=argparse.FileType('w'), required=False,
                                 help='Target CSV file where to dump data.')
    parser.add_argument('--binaries', required=False, default=None,
                                 help='Specify non TaskCluster native_client.tar.xz to use')
    return parser.parse_args()

def do_main():
    cli_args = handle_args()

    if not cli_args.models or not cli_args.wav or not cli_args.alphabet:
        raise AssertionError('Missing arguments (models, wav or alphabet)')

    if cli_args.so_model:
        '''
        Verify we have a string that matches the format described in
        reduce_filename above: NAME.aot.EPOCHS.XXX.YYY.so
         - Where XXX is a variation on the model size for example
         - And where YYY is a const related to the training dataset
        '''

        parts = cli_args.so_model.split('.')
        assert len(parts) == 6
        assert parts[1]   == 'aot'
        assert parts[-1]  == 'so'

    if cli_args.dir is not None and not os.path.isdir(cli_args.dir):
        raise AssertionError('Inexistent temp directory')

    if cli_args.binaries is not None and cli_args.binaries.find('native_client.tar.xz') == -1:
        raise AssertionError('Local binaries must be bundled in a native_client.tar.xz file')

    global ssh_conn
    ssh_conn = establish_ssh(target=cli_args.target, auto_trust=cli_args.autotrust, allow_agent=cli_args.allowagent, look_keys=cli_args.lookforkeys)

    tempdir, sorted_models = setup_tempdir(dir=cli_args.dir, models=cli_args.models, wav=cli_args.wav, alphabet=cli_args.alphabet, lm_binary=cli_args.lm_binary, trie=cli_args.trie, binaries=cli_args.binaries)

    dest_sorted_models = list(map(lambda x: os.path.join(tempdir, os.path.basename(x)), sorted_models))
    dest_wav = os.path.join(tempdir, os.path.basename(cli_args.wav))
    dest_alphabet = os.path.join(tempdir, os.path.basename(cli_args.alphabet))

    if cli_args.lm_binary and cli_args.trie:
        dest_lm_binary = os.path.join(tempdir, os.path.basename(cli_args.lm_binary))
        dest_trie = os.path.join(tempdir, os.path.basename(cli_args.trie))
        inference_times = run_benchmarks(dir=tempdir, models=dest_sorted_models, extra_aot_model=cli_args.so_model, wav=dest_wav, alphabet=dest_alphabet, lm_binary=dest_lm_binary, trie=dest_trie, iters=cli_args.iters)
    else:
        inference_times = run_benchmarks(dir=tempdir, models=dest_sorted_models, extra_aot_model=cli_args.so_model, wav=dest_wav, alphabet=dest_alphabet, iters=cli_args.iters)

    if cli_args.csv:
        produce_csv(input=inference_times, output=cli_args.csv)

    if not cli_args.keep:
        teardown_tempdir(dir=tempdir)

if __name__ == '__main__' :
    do_main()
