#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import os
import sys
# To use util.tc
sys.path.append(os.path.abspath('.'))

import paramiko
import argparse
import tempfile
import shutil
import subprocess
import stat
import numpy
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
import scipy.io.wavfile as wav
import csv
import getpass
import zipfile

import util.tc as tcu

r'''
 Tool to:
  - remote local or remote (ssh) native_client
  - handles copying models (as protocolbuffer files)
  - run native_client in benchmark mode
  - collect timing results
  - compute mean values (with wariances)
  - output as CSV and plots
'''

TASKCLUSTER_SCHEME = 'https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.master.%(arch_string)s/artifacts/public/native_client.tar.xz'

ssh_conn = None
def exec_command(command=None, cwd=None):
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

def assert_valid_dir(dir=None):
    if dir is None:
        raise AssertionError('Valid temp directory')
    return True

def get_arch_string():
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

def get_tc_url(arch_string=None):
    return TASKCLUSTER_SCHEME % { 'arch_string': arch_string }

def maybe_download_binaries(dir=None):
    assert_valid_dir(dir)
    tcu.maybe_download_tc(target_dir=dir, tc_url=get_tc_url(get_arch_string()), progress=True)

def extract_tarball(dir=None):
    assert_valid_dir(dir)

    target_tarball = os.path.join(dir, 'native_client.tar.xz')
    if os.path.isfile(target_tarball) and os.stat(target_tarball).st_size == 0:
        return

    subprocess.check_call(['pixz', '-d', 'native_client.tar.xz'], cwd=dir)
    subprocess.check_call(['tar', 'xf', 'native_client.tar'], cwd=dir)
    os.unlink(os.path.join(dir, 'native_client.tar'))
    open(target_tarball, 'w').close()

def maybe_inspect_zip(models=None):
    r'''
    Detect if models is a list of protocolbuffer files or a ZIP file.
    If the latter, then unzip it and return the list of protocolbuffer files
    that were inside.
    '''
    assert_valid_dir(dir)

    if len(models) > 1:
        return models

    if len(models) < 1:
        raise AssertionError('No models at all')

    return zipfile.ZipFile(models[0]).namelist()

def keep_only_digits(s):
    r'''
    local helper to just keep digits
    '''
    fs = ''
    for c in s:
        if ord(c) >= ord('0') and ord(c) <= ord('9'):
            fs += c

    return int(fs)

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

    base = map(lambda x: os.path.abspath(x), maybe_inspect_zip(models))
    base.sort(cmp=nsort)

    return base

def copy_tree(dir=None):
    assert_valid_dir(dir)

    sftp = ssh_conn.open_sftp()
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
        #else:
        #    print('File exists: %s' % fullpath)

        if local_stat.st_mode != remote_mode:
            print('Setting proper remote mode: %s' % local_stat.st_mode)
            sftp.chmod(fullpath, local_stat.st_mode)

    sftp.close()

def delete_tree(dir=None):
    assert_valid_dir(dir)

    sftp = ssh_conn.open_sftp()
    if not stat.S_ISDIR(sftp.stat(dir).st_mode):
        print('No remote directory: %s' % dir)
    else:
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

def setup_tempdir(dir=None, models=[], wav=None, binaries=None):
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
    extract_tarball(dir)

    filenames = map(lambda x: os.path.join(dir, os.path.basename(x)), sorted_models)
    missing_models = filter(lambda x: not os.path.isfile(x), filenames)
    if len(missing_models) > 0:
        # If we have a ZIP file, directly extract it to the proper path
        if len(models) == 1:
            print('Extracting %s to %s' % (models[0], dir))
            zipfile.ZipFile(models[0]).extractall(path=dir)
            print('Extracted %s.' % models[0])
        else:
            # If one model is missing, let's copy everything again. Be safe.
            for f in sorted_models:
                print('Copying %s to %s' % (f, dir))
                shutil.copy2(f, dir)

    if not os.path.isfile(os.path.join(dir, os.path.basename(wav))):
        print('Copying %s to %s' % (wav, dir))
        shutil.copy2(wav, dir)

    if ssh_conn:
        copy_tree(dir)

    return dir, sorted_models

def teardown_tempdir(dir=None, keep=False):
    if keep:
        return

    if ssh_conn:
        delete_tree(dir)

    assert_valid_dir(dir)
    shutil.rmtree(dir)

def get_sshconfig():
    with open(os.path.expanduser('~/.ssh/config')) as f:
        cfg = paramiko.SSHConfig()
        cfg.parse(f)
        ret_dict = {}
        for d in cfg._config:
            _copy = dict(d)
            del _copy['host']
            for host in d['host']:
                ret_dict[host] = _copy['config']

        return ret_dict

def establish_ssh(target=None, auto_trust=False, allow_agent=True, look_keys=True):
    def password_prompt(username, hostname):
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

            if 'identityfile' in user_config:
                cfg['key_filename'] = user_config['identityfile']
            else:
                cfg['password'] = password_prompt(cfg['username'], cfg['hostname'] or target)

            # Should be the last one, since it will issue connection to remote host
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

def run_benchmarks(dir=None, models=None, wav=None, iters=-1):
    assert_valid_dir(dir)

    inference_times = [ ]

    for model in models:
        current_model = {
          'name':   model,
          'iters':  [ ],
          'mean':   numpy.infty,
          'stddev': numpy.infty
        }

        cmdline = './deepspeech %s %s -t' % (model, wav)
        for it in range(iters):
            sys.stdout.write("\rRunning %s: %d/%d" % (os.path.basename(model), (it+1), iters))
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

        sys.stdout.write('\n')
        sys.stdout.flush()
        current_model['mean']   = numpy.mean(current_model['iters'])
        current_model['stddev'] = numpy.std(current_model['iters'])
        inference_times.append(current_model)

    return inference_times

def produce_csv(input=None, output=None):
    output.write('"model","mean","std"\n')
    for model_data in input:
        output.write('"%s",%f,%f\n' % (model_data['name'], model_data['mean'], model_data['stddev']))
    output.flush()
    output.close()
    print("Wrote as %s" % output.name)

def ingest_csv(datasets=None, range=None):
    existing_files = filter(lambda x: os.path.isfile(x[1]), datasets)
    assert len(datasets) == len(existing_files)

    if range:
        range = map(lambda x: int(x), range.split(','))

    datas = {}
    for (dsname, dsfile) in datasets:
        print('Reading %s from %s' % (dsname, dsfile))
        with open(dsfile) as f:
            d = csv.DictReader(f)
            datas[dsname] = []
            for e in d:
                if range:
                    re       = reduce_filename(e['model'])
                    in_range = (re >= range[0] and re <= range[1])
                    if in_range:
                        datas[dsname].append(e)
                else:
                    datas[dsname].append(e)

    return datas

def produce_plot(input=None, output=None):
    x = range(len(input))
    xlabels = map(lambda a: a['name'], input)
    y = map(lambda a: a['mean'], input)
    yerr = map(lambda a: a['stddev'], input)

    print('y=', y)
    print('yerr=', yerr)
    plt.errorbar(x, y, yerr=yerr)
    plt.show()
    print("Wrote as %s" % output.name)

def reduce_filename(f):
    r'''
    Expects something like /tmp/tmpAjry4Gdsbench/test.weights.e5.XXX.YYY.pb
    Where XXX is a variation on the model size for example
    And where YYY is a const related to the training dataset
    '''

    f = os.path.basename(f).split('.')

    return keep_only_digits(f[-3])

def produce_plot_multiseries(input=None, output=None, title=None, size=None, fig_dpi=None, source_wav=None):
    fig, ax = plt.subplots()
    fig.set_figwidth(float(size.split('x')[0]) / fig_dpi)
    fig.set_figheight(float(size.split('x')[1]) / fig_dpi)

    nb_items = len(input[input.keys()[0]])
    x_all    = range(nb_items)
    for serie in input.keys():
        serie_values = input[serie]

        xtics  = map(lambda a: reduce_filename(a['model']), serie_values)
        y      = map(lambda a: float(a['mean']), serie_values)
        yerr   = map(lambda a: float(a['std']), serie_values)
        linreg = scipy_stats.linregress(x_all, y)
        ylin   = linreg.intercept + linreg.slope * numpy.asarray(x_all)

        ax.errorbar(x_all, y, yerr=yerr, label=('%s' % serie), fmt='-', capsize=4, elinewidth=1)
        ax.plot(x_all, ylin, label=('%s ~= %0.4f*x+%0.4f (R=%0.4f)' % (serie, linreg.slope, linreg.intercept, linreg.rvalue)))

        plt.xticks(x_all, xtics, rotation=60)

    if source_wav:
        audio = wav.read(source_wav)
        print('Adding realtime')
        for rt_factor in [ 1.0, 1.5, 2.0 ]:
            rt_secs = float(len(audio[1])) / float(1.0 * audio[0]) * rt_factor
            y_rt    = numpy.repeat(rt_secs, nb_items)
            ax.plot(x_all, y_rt, label=('Realtime: %0.4f secs [%0.1f]' % (rt_secs, rt_factor)))

    ax.set_title(title)
    ax.set_xlabel('Model size')
    ax.set_ylabel('Execution time (s)')
    legend = ax.legend(loc='best')

    plt.grid()
    plt.tight_layout()
    plt.savefig(output, transparent=False, frameon=True, dpi=fig_dpi)

def run_as_benchmark(args=None):
    if not args:
        raise AssertionError('Unexpected state')

    if args.dir and os.path.isdir(args.dir) and args.models is not None and args.wav is not None:
        return True

    return False

def handle_args():
    parser = argparse.ArgumentParser(description='Benchmarking tooling for DeepSpeech native_client.')
    benchmark_group = parser.add_argument_group(title='bench', description='Benchmarking options')
    benchmark_group.add_argument('--target', type=str, required=False,
                                 help='SSH user:pass@host string for remote benchmarking.')
    benchmark_group.add_argument('--autotrust', action='store_true', default=False,
                                 help='SSH Paramiko policy to automatically trust unknown keys.')
    benchmark_group.add_argument('--allowagent', action='store_true', dest='allowagent',
                                 help='Allow the use of a SSH agent.')
    benchmark_group.add_argument('--no-allowagent', action='store_false', dest='allowagent',
                                 help='Disallow the use of a SSH agent.')
    benchmark_group.add_argument('--lookforkeys', action='store_true', dest='lookforkeys',
                                 help='Allow to look for SSH keys in ~/.ssh/.')
    benchmark_group.add_argument('--no-lookforkeys', action='store_false', dest='lookforkeys',
                                 help='Disallow to look for SSH keys in ~/.ssh/.')
    benchmark_group.add_argument('--dir', type=str, required=False, default=None,
                                 help='Local directory where to copy stuff.')
    benchmark_group.add_argument('--models', type=str, nargs='+', required=False,
                                 help='List of files (protocolbuffer) to work on. Might be a zip file.')
    benchmark_group.add_argument('--wav', type=str, required=False,
                                 help='WAV file to pass to native_client. Supply again in plotting mode to draw realine line.')
    benchmark_group.add_argument('--iters', type=int, required=False, default=5,
                                 help='How many iterations to perfom on each model.')
    benchmark_group.add_argument('--keep', required=False, action='store_true',
                                 help='Keeping run files (binaries & models).')
    benchmark_group.add_argument('--csv', type=argparse.FileType('w'), required=False,
                                 help='Target CSV file where to dump data.')
    benchmark_group.add_argument('--binaries', type=str, required=False, default=None,
                                 help='Specify non TaskCluster native_client.tar.xz to use')
    plotting_group = parser.add_argument_group(title='plot', description='Plotting options')
    plotting_group.add_argument('--dataset', action='append', nargs=2, metavar=('name','source'),
                                help='Include dataset NAME from file SOURCE. Repeat the option to add more datasets.')
    plotting_group.add_argument('--title', type=str, default=None,
                                help='Title of the plot.')
    plotting_group.add_argument('--plot', type=argparse.FileType('w'), required=False,
                                help='Target file where to plot data. Format will be deduced from extension.')
    plotting_group.add_argument('--size', type=str, default='800x600',
                                help='Size (px) of the resulting plot.')
    plotting_group.add_argument('--dpi', type=int, default=96,
                                help='Set plot DPI.')
    plotting_group.add_argument('--range', type=str, default=None,
                                help='Range of model size to use. Comma-separated string of boundaries: min,max')
    return parser.parse_args()

def do_main():
    cli_args = handle_args()

    if run_as_benchmark(cli_args):
        print('Running as benchmark')

        if not cli_args.models or not cli_args.wav:
            raise AssertionError('Missing arguments (models or wav)')

        if cli_args.dir is not None and not os.path.isdir(cli_args.dir):
            raise AssertionError('Inexistent temp directory')

        if cli_args.binaries is not None and cli_args.binaries.find('native_client.tar.xz') == -1:
            raise AssertionError('Local binaries must be bundled in a native_client.tar.xz file')

        global ssh_conn
        ssh_conn = establish_ssh(target=cli_args.target, auto_trust=cli_args.autotrust, allow_agent=cli_args.allowagent, look_keys=cli_args.lookforkeys)

        tempdir, sorted_models = setup_tempdir(dir=cli_args.dir, models=cli_args.models, wav=cli_args.wav, binaries=cli_args.binaries)

        dest_sorted_models = map(lambda x: os.path.join(tempdir, os.path.basename(x)), sorted_models)
        dest_wav = os.path.join(tempdir, os.path.basename(cli_args.wav))

        inference_times = run_benchmarks(dir=tempdir, models=dest_sorted_models, wav=dest_wav, iters=cli_args.iters)
        if cli_args.csv:
            produce_csv(input=inference_times, output=cli_args.csv)

        teardown_tempdir(dir=tempdir, keep=cli_args.keep)
    else:
        print('Running as plotter')
        all_inference_times = ingest_csv(datasets=cli_args.dataset, range=cli_args.range)

        if cli_args.plot:
            produce_plot_multiseries(input=all_inference_times, output=cli_args.plot, title=cli_args.title, size=cli_args.size, fig_dpi=cli_args.dpi, source_wav=cli_args.wav)

if __name__ == '__main__' :
    do_main()
