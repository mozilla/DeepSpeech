# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
import json
import os
import git
import requests
import sys
import shutil
import subprocess
import datetime
import copy
import csv

from glob import glob
from xdg import BaseDirectory
from threading import Thread
from time import time
from scipy.interpolate import spline

from six.moves import range
# Do this to be able to use without X
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GITHUB_API_BASE        = 'https://api.github.com'
MOZILLA_GITHUB_ACCOUNT = os.environ.get('ds_github_account', 'mozilla')
DEEPSPEECH_GITHUB_PROJ = os.environ.get('ds_github_project', 'DeepSpeech')
DEEPSPEECH_GITHUB_REF  = os.environ.get('ds_github_ref',     'refs/heads/wer-tracking')

DEEPSPEECH_CLONE_PATH  = os.path.abspath(os.environ.get('ds_clone_path', './ds_exec_clone/'))

CACHE_DIR = os.path.abspath(os.path.join(BaseDirectory.xdg_cache_home, 'deepspeech_wer'))
DATA_DIR  = os.path.abspath(os.path.join(os.environ.get('ds_dataroot', BaseDirectory.xdg_data_home), 'deepspeech_wer'))

# Lock file to prevent execution: avoid multiple simultaneous execution, blocks
# execution if needed
LOCKFILE  = os.path.join(CACHE_DIR, 'lock')

# File holding the last SHA1 we processed
SHA1FILE  = os.path.join(CACHE_DIR, 'last_sha1')

# Checkpoint to force restore, should point to a valid checkpoint directory
CKPTFILE  = os.path.join(CACHE_DIR, 'checkpoint_file')

# Checkpoint base dir, will output in subdir based on git sha1
CKPT_BASE_DIR  = os.path.join(DATA_DIR, 'checkpoint')

class GPUUsage(Thread):
    def __init__(self, csvfile=None):
        super(GPUUsage, self).__init__()

        self._cmd        = [ 'nvidia-smi', 'dmon', '-d', '1', '-s', 'pucvmet' ]
        self._names      = []
        self._units      = []
        self._process    = None

        self._csv_output = csvfile or os.environ.get('ds_gpu_usage_csv', self.make_basename(prefix='ds-gpu-usage', extension='csv'))

    def make_basename(self, prefix, extension):
        # Let us assume that this code is executed in the current git clone
        return '%s.%s.%s.%s' % (prefix, git.Repo('.').git.describe('--always', '--dirty', '--abbrev'), int(time()), extension)

    def stop(self):
        if not self._process:
            print("Trying to stop nvidia-smi but no more process, please fix.")
            return

        print("Ending nvidia-smi monitoring: PID", self._process.pid)
        self._process.terminate()
        print("Ended nvidia-smi monitoring ...")

    def run(self):
        print("Starting nvidia-smi monitoring")

        # If the system has no CUDA setup, then this will fail.
        try:
            self._process = subprocess.Popen(self._cmd, stdout=subprocess.PIPE)
        except OSError as ex:
            print("Unable to start monitoring, check your environment:", ex)
            return

        writer = None
        with open(self._csv_output, 'w') as f:
            for line in iter(self._process.stdout.readline, ''):
                d = self.ingest(line)

                if line.startswith('# '):
                    if len(self._names) == 0:
                        self._names = d
                        writer = csv.DictWriter(f, delimiter=str(','), quotechar=str('"'), fieldnames=d)
                        writer.writeheader()
                        continue
                    if len(self._units) == 0:
                        self._units = d
                        continue
                else:
                    assert len(self._names) == len(self._units)
                    assert len(d) == len(self._names)
                    assert len(d) > 1
                    writer.writerow(self.merge_line(d))
                    f.flush()

    def ingest(self, line):
        return map(lambda x: x.replace('-', '0'), filter(lambda x: len(x) > 0, map(lambda x: x.strip(), line.split(' ')[1:])))

    def merge_line(self, line):
        return dict(zip(self._names, line))

class GPUUsageChart():
    def __init__(self, source, basename=None):
        self._rows    = [ 'pwr', 'temp', 'sm', 'mem']
        self._titles  = {
            'pwr':  "Power (W)",
            'temp': "Temperature (°C)",
            'sm':   "Streaming Multiprocessors (%)",
            'mem':  "Memory (%)"
        }
        self._data     = { }.fromkeys(self._rows)
        self._csv      = source
        self._basename = basename or os.environ.get('ds_gpu_usage_charts', 'gpu_usage_%%s_%d.png' % int(time.time()))

        # This should make sure we start from anything clean.
        plt.close("all")

        try:
            self.read()
            for plot in self._rows:
                self.produce_plot(plot)
        except IOError as ex:
            print("Unable to read", ex)

    def append_data(self, row):
        for bucket, value in row.iteritems():
            if not bucket in self._rows:
                continue

            if not self._data[bucket]:
                self._data[bucket] = {}

            gpu = int(row['gpu'])
            if not self._data[bucket].has_key(gpu):
                self._data[bucket][gpu]  = [ value ]
            else:
                self._data[bucket][gpu] += [ value ]

    def read(self):
        print("Reading data from", self._csv)
        with open(self._csv, 'r') as f:
            for r in csv.DictReader(f):
                self.append_data(r)

    def produce_plot(self, key, with_spline=True):
        png = self._basename % (key, )
        print("Producing plot for", key, "as", png)
        fig, axis = plt.subplots()
        data = self._data[key]
        if data is None:
            print("Data was empty, aborting")
            return

        x = list(range(len(data[0])))
        if with_spline:
            x = map(lambda x: float(x), x)
            x_sm = np.array(x)
            x_smooth = np.linspace(x_sm.min(), x_sm.max(), 300)

        for gpu, y in data.iteritems():
            if with_spline:
                y = map(lambda x: float(x), y)
                y_sm = np.array(y)
                y_smooth = spline(x, y, x_smooth, order=1)
                axis.plot(x_smooth, y_smooth, label='GPU %d' % (gpu))
            else:
                axis.plot(x, y, label='GPU %d' % (gpu))

        axis.legend(loc="upper right", frameon=False)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("%s" % self._titles[key])
        fig.set_size_inches(24, 18)
        plt.title("GPU Usage: %s" % self._titles[key])
        plt.savefig(png, dpi=100)
        plt.close(fig)

def try_get_lock():
    """
    Try to acquire a lock using a file-based mechanism
    Will raise Exception('lock') when cannot acquire lock
    Please release with release_lock()
    """

    # Let us create empty file in case nothing exists
    if not os.path.isdir(os.path.dirname(LOCKFILE)):
        os.makedirs(os.path.dirname(LOCKFILE))

    # No pre-existing lock, take one
    if not os.path.isfile(LOCKFILE):
        with open(LOCKFILE, 'w') as fd:
            fd.write('%d' % os.getpid())
    else:
        raise Exception('lock')

def release_lock():
    """
    Release the previously aqcuired lock, assume file exists
    It will verify lock was acquired by the same process, or otherwise it will
    raise Exception('pid')
    """

    with open(LOCKFILE, 'r') as fd:
        lock_pid = int(fd.read().strip())
        curr_pid = int(os.getpid())
        if lock_pid != curr_pid:
            raise Exception('pid')

    # If everything went fine, just release the lock
    os.unlink(LOCKFILE)

def sys_exit_safe(rcode=1):
    """
    A sys.exit(1) that releases the lock
    """

    release_lock()
    sys.exit(rcode)

def get_last_sha1():
    # Let us create empty file in case nothing exists
    if not os.path.isfile(SHA1FILE):
        with open(SHA1FILE, 'w') as fd:
            fd.write('')

    with open(SHA1FILE, 'r') as fdr:
        content = fdr.read().strip()

    return content

def write_last_sha1(sha):
    with open(SHA1FILE, 'w') as fd:
        fd.write(sha)

def get_github_repo_url():
    """
    Build URL to github repo
    """
    return 'git://github.com/%s/%s.git' % (MOZILLA_GITHUB_ACCOUNT, DEEPSPEECH_GITHUB_PROJ)

def get_github_ref_url():
    """
    Fetches the sha1 ref for the current refs/heads/master
    """
    return '%s/repos/%s/%s/git/%s' % (GITHUB_API_BASE, MOZILLA_GITHUB_ACCOUNT, DEEPSPEECH_GITHUB_PROJ, DEEPSPEECH_GITHUB_REF)

def get_github_compare_url(last_sha1):
    """
    Build the git log last_sha1...refs/heads/master comparison URL, to check if
    the current recorded last_sha1 is identical to remote's head or if the
    refs/heads/master is ahead of the last_sha1
    """
    return '%s/repos/%s/%s/compare/%s...%s' % (GITHUB_API_BASE, MOZILLA_GITHUB_ACCOUNT, DEEPSPEECH_GITHUB_PROJ, last_sha1, DEEPSPEECH_GITHUB_REF)

def git_date(d):
    """
    Returns datetime object from a git date
    """
    return datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%SZ')

def is_webflow(commit):
    """
    Check if commit is a valid "webflow" commit, i.e., whose committer is web-flow
    """
    return commit['committer'] and (commit['committer']['login'] == 'web-flow')

def is_newer(refdate, commit):
    """
    Check is a merge commit is a GitHub newer one
    """
    c = commit['commit']['committer']
    return c and (c['name'] == 'GitHub') and (git_date(c['date']) >= refdate)

def get_current_sha1():
    r = requests.get(get_github_ref_url())
    if r.status_code is not 200:
        return None, r.status_code
    payload = json.loads(r.text)
    return payload['object']['sha'], r.status_code

def get_new_commits(sha1_from):
    r = requests.get(get_github_compare_url(sha1_from))
    if r.status_code is not 200:
        return None, r.status_code
    payload = json.loads(r.text)

    this_merge_date = git_date(payload['base_commit']['commit']['committer']['date'])

    if payload['status'] == 'identical':
        # When there is no change, just nicely output no new commits
        return [], r.status_code
    elif payload['status'] == 'ahead':
        # Let us just keep the merges commits, we can identify them because they
        # are the commits with two parents
        all_merges = filter(lambda x: len(x['parents']) >= 2, payload['commits'])
        # Keep only merges made from Github UI
        gh_merges = filter(lambda x: is_webflow(x), all_merges)
        # Keep only merges with committer date >= this_merge_date
        new_merges = filter(lambda x: is_newer(this_merge_date, x), gh_merges)
        return map(lambda x: x['sha'], new_merges), r.status_code

    # Should not happen
    return None, r.status_code

def get_git_repo_path():
    """
    Build the path to git repo
    """
    return os.path.join(DEEPSPEECH_CLONE_PATH, '.git')

def get_git_desc():
    """
    Produce git --describe --always --abbrev --dirty from git repo
    """

    gitdir = get_git_repo_path()
    assert os.path.isdir(gitdir)

    return git.Repo(gitdir).git.describe('--always', '--dirty', '--abbrev')

def ensure_git_clone(sha):
    """
    Ensure that there is an existing git clone of the root repo and make sure we
    checkout at `sha`.
    Clone will be made to the path designated by DEEPSPEECH_CLONE_PATH.
    We will wipe everything after the run(s).
    """

    gitdir = get_git_repo_path()
    # If non-existent, create a clone
    if not os.path.isdir(gitdir):
        source = get_github_repo_url()
        print("Performing a fresh clone from", source, "to", DEEPSPEECH_CLONE_PATH)
        ds_repo = git.Repo.clone_from(source, DEEPSPEECH_CLONE_PATH)
    else:
        print("Using existing clone from", DEEPSPEECH_CLONE_PATH)
        ds_repo = git.Repo(gitdir)

    # Ensure we have a valid non bare local clone
    assert ds_repo.__class__ is git.Repo
    assert not ds_repo.bare

    commit = ds_repo.commit(sha)
    assert commit.__class__ is git.Commit

    # Detach HEAD to the SHA1. GitPython doc says this is dangerous, but we want
    # to make sure we obliterate anything.
    ds_repo.head.reference = commit
    ds_repo.head.reset(index=True, working_tree=True)

    assert ds_repo.head.is_detached

    return True

def wipe_git_clone():
    """
    Eradicate the git clone we made earlier
    """
    shutil.rmtree(DEEPSPEECH_CLONE_PATH)

def ensure_gpu_usage(root_dir):
    """
    Prepare storing of GPU usage CSV file and PNG charts. Will use the directory
    specified within ds_gpu_usage_root as a root.

    Returns two strings to build env
    """

    gpu_usage_root = os.path.abspath(os.environ.get('ds_gpu_usage_root', root_dir))
    gpu_usage_path = os.path.join(gpu_usage_root, get_git_desc())

    print("Will produce CSV and charts in %s" % gpu_usage_path)
    if not os.path.isdir(gpu_usage_path):
        os.makedirs(gpu_usage_path)

    rundate = int(time())

    ds_gpu_usage_csv    = os.path.join(gpu_usage_path, 'gpu-usage-%d.csv' % rundate)
    ds_gpu_usage_charts = os.path.join(gpu_usage_path, 'gpu-usage-%d-%%s.png' % rundate)

    return ds_gpu_usage_csv, ds_gpu_usage_charts

def ensure_checkpoint_directory():
    """
    Take care of handling checkpoint directory:
      - setting proper value
      - handling of force restore file
    """

    # Defaulting to re-using checkpoint from the same SHA1 directory, but setting
    # the checkpoint_restore boolean only of the directory actually exists
    # If it exists but it is empty, then notebook will likely fail.
    checkpoint_dir     = os.path.join(CKPT_BASE_DIR, get_git_desc())

    # Check if a force restore checkpoint file exists
    if os.path.isfile(CKPTFILE):
        with open(CKPTFILE, 'r') as ckpt_file:
            maybe_checkpoint_dir = ckpt_file.read().strip()

        # This file is intended to be a one-time use, let us remove it whatever
        # happens after. It is up to the user to re-set it. This is to avoid
        # restarting from a past checkpoint if not explicitely done on purpose.
        os.unlink(CKPTFILE)

        # Make sure the targetted path actually exists.
        # We check here to not change the value of checkpoint_dir in case
        # it does not exists.
        # If it actually exists, the later os.path.isdir(checkpoint_dir) will
        # take care of setting the checkpoint_restore value
        print("Trying with maybe_checkpoint_dir", maybe_checkpoint_dir)
        if not os.path.isdir(maybe_checkpoint_dir):
            print("Trying to force-restore checkpoint from non-existent directory", maybe_checkpoint_dir)
        else:
            checkpoint_dir = maybe_checkpoint_dir

    # Create if non existent
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        print("Found existing checkpoint dir, re-using it")

    print("Ensured checkpoint infos: ", checkpoint_dir)

    return checkpoint_dir

def exec_wer_run():
    """
    Execute one run, blocking for the completion of the process. We default
    to ./bin/run-wer-automation.sh
    """

    ds_script = os.environ.get('ds_wer_automation', './bin/run-wer-automation.sh')
    if not os.path.isfile(ds_script):
        raise Exception('invalid-script')

    # Collect expected checkpoint directory depending on what the user wants
    # defaulting to re-using checkpoint from the same SHA1
    ckpt_dir = ensure_checkpoint_directory()

    # Copy current process environment to be able to augment it
    local_env = copy.deepcopy(os.environ)

    # Pass the current environment, it is required for user to supply the upload
    # informations used by the notebook, and it also makes us able to run all
    # this within loaded virtualenv
    res = subprocess.check_call([ ds_script, '--checkpoint_dir', ckpt_dir ], env=local_env)

    assert res == 0

##########################
#       FOLLOW ME        #
# TO A PARALLEL UNIVERSE #
##########################

def exec_main():
    """
    This is the main function. We isolate it to be able to load automation as a
    module from other places
    """

    try_get_lock()

    # Allow user to force a SHA1 from env, mostly for testing purpose without
    # putting pressure on Github API (anon is rate-limited to 60 reqs/hr from the
    # same IP address.
    try:
        sha_to_run = os.environ.get('ds_automation_force_sha').split(',')
    except:
        sha_to_run = [ ]

    # ensure we have a real dir name
    assert len(DEEPSPEECH_CLONE_PATH) > 3
    assert os.path.isabs(DEEPSPEECH_CLONE_PATH)

    # Creating holding directories if needed
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    if len(sha_to_run) == 0:
        current = get_last_sha1()
        if len(current) == 40:
            print('Existing SHA1:', current, 'fetching changes')
            sha_to_run, rt = get_new_commits(current)
            if sha_to_run is not None and len(sha_to_run) is 0:
                print("No new SHA1, got HTTP status:", rt)
                sys_exit_safe()
            elif sha_to_run is None:
                print("Something went badly wrong, unable to use Github compare")
                sys_exit_safe()
        else:
            # Ok, we do not have an existing SHA1, let us get one
            print('No pre-existing SHA1, fetching refs')
            sha1, rt = get_current_sha1()
            if sha1 is None:
                print("No SHA1, got HTTP status:", rt)
                sys_exit_safe()
            sha_to_run = [ sha1 ]
    else:
        print("Using forced SHA1 from env")

    print("Will execute for", sha_to_run)

    for sha in sha_to_run:
        if not ensure_git_clone(sha):
            print("Error with git repo handling.")
            sys_exit_safe()

        print("Ready for", sha)

        print("Let us place ourselves into the git clone directory ...")
        root_dir = os.getcwd()
        os.chdir(DEEPSPEECH_CLONE_PATH)

        print("Starting GPU nvidia-smi monitoring")
        gpu_usage_csv, gpu_usage_charts = ensure_gpu_usage(root_dir)
        gu = GPUUsage(csvfile=gpu_usage_csv)
        gu.start()

        print("Do the training for getting WER computation")
        exec_wer_run()

        print("Producing GPU monitoring charts")
        gu.stop()
        GPUUsageChart(source=gpu_usage_csv, basename=gpu_usage_charts)

        print("Save progress")
        write_last_sha1(sha)

        print("Let us place back to the previous directory %s ..." % root_dir)
        os.chdir(root_dir)

    print("Getting rid of git clone")
    wipe_git_clone()

    release_lock()

# Only execute when triggered directly. If loaded as a module don't do this.
if __name__ == "__main__":
    exec_main()
