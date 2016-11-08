import json
import os
import git
import requests
import sys
import shutil
import subprocess
import datetime

from glob import glob
from xdg import BaseDirectory

GITHUB_API_BASE        = 'https://api.github.com'
MOZILLA_GITHUB_ACCOUNT = os.environ.get('ds_github_account', 'mozilla')
DEEPSPEECH_GITHUB_PROJ = os.environ.get('ds_github_project', 'DeepSpeech')
DEEPSPEECH_GITHUB_REF  = os.environ.get('ds_github_ref',     'refs/heads/wer-tracking')

DEEPSPEECH_CLONE_PATH  = os.path.abspath(os.environ.get('ds_clone_path', './ds_exec_clone/'))

CACHE_DIR = os.path.join(BaseDirectory.xdg_cache_home, 'deepspeech_wer')
LOCKFILE  = os.path.join(CACHE_DIR, 'lock')
SHA1FILE  = os.path.join(CACHE_DIR, 'last_sha1')

DATA_DIR  = os.path.join(BaseDirectory.xdg_data_home, 'deepspeech_wer')
CKPT_DIR  = os.path.join(DATA_DIR, 'checkpoint')

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
        if not os.path.isdir(CACHE_DIR):
            os.makedirs(CACHE_DIR)
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

def ensure_git_clone(sha):
    """
    Ensure that there is an existing git clone of the root repo and make sure we
    checkout at `sha`.
    Clone will be made to the path designated by DEEPSPEECH_CLONE_PATH.
    We will wipe everything after the run(s).
    """

    gitdir = os.path.join(DEEPSPEECH_CLONE_PATH, '.git')

    # If non-existent, create a clone
    if not os.path.isdir(gitdir):
        source = get_github_repo_url()
        print "Performing a fresh clone from", source, "to", DEEPSPEECH_CLONE_PATH
        ds_repo = git.Repo.clone_from(source, DEEPSPEECH_CLONE_PATH)
    else:
        print "Using existing clone from", DEEPSPEECH_CLONE_PATH
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

def populate_previous_logs():
    """
    Let us copy previous logs produced by runs from the copy we are keeping in
    $XDG_DATA_DIR, into the git clone rep (DEEPSPEECH_CLONE_PATH).
    If the target directory exists, it means we probably already have everything
    """

    # Creating holding directory if needed
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    src_logs_dir = os.path.join(DATA_DIR, 'logs')
    dst_logs_dir = os.path.join(DEEPSPEECH_CLONE_PATH, 'logs')
    if not os.path.isdir(src_logs_dir):
        # There is nothing to copy back, let's just bail out
        return

    if os.path.isdir(dst_logs_dir):
        # Directory already exists, just bail out
        return

    # We assume the content is sane and just copy everything
    shutil.copytree(src_logs_dir, dst_logs_dir)

def save_logs():
    """
    Copy the logs that have been produced by the WER run to the backup location
    We will just take hyper.json files
    """
    # Creating holding directory if needed
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    src_logs_dir = os.path.join(DEEPSPEECH_CLONE_PATH, 'logs')
    glob_mask    = '%s/*/hyper.json' % src_logs_dir
    for hyperJson in map(lambda x: os.path.relpath(x, DEEPSPEECH_CLONE_PATH), glob(glob_mask)):
        final_path = os.path.join(DATA_DIR, hyperJson)
        if not os.path.isdir(os.path.dirname(final_path)):
            os.makedirs(os.path.dirname(final_path))
        shutil.copy(hyperJson, final_path)

def exec_wer_run():
    """
    Execute one run, blocking for the completion of the process. We default
    to ./bin/run-wer-automation.sh
    We also create a clean checkpoint directory in local user dir
    """

    # Creating holding directory if needed
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Remove if existing
    if os.path.isdir(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)

    # Create if non existent
    if not os.path.isdir(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    ds_script = os.environ.get('ds_wer_automation', './bin/run-wer-automation.sh')
    if not os.path.isfile(ds_script):
        raise Exception('invalid-script')

    # Force automation to use a user-local checkpoint dir
    local_env = os.environ.update({'ds_checkpoint_dir': CKPT_DIR})

    # Pass the current environment, it is required for user to supply the upload
    # informations used by the notebook, and it also makes us able to run all
    # this within loaded virtualenv
    res = subprocess.check_call([ ds_script ], env=local_env)

    assert res == 0

##########################
#       FOLLOW ME        #
# TO A PARALLEL UNIVERSE #
##########################

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

if len(sha_to_run) == 0:
    current = get_last_sha1()
    if len(current) == 40:
        print 'Existing SHA1:', current, 'fetching changes'
        sha_to_run, rt = get_new_commits(current)
        if sha_to_run is not None and len(sha_to_run) is 0:
            print "No new SHA1, got HTTP status:", rt
            sys_exit_safe()
        elif sha_to_run is None:
            print "Something went badly wrong, unable to use Github compare"
            sys_exit_safe()
    else:
        # Ok, we do not have an existing SHA1, let us get one
        print 'No pre-existing SHA1, fetching refs'
        sha1, rt = get_current_sha1()
        if sha1 is None:
            print "No SHA1, got HTTP status:", rt
            sys_exit_safe()
        sha_to_run = [ sha1 ]
else:
    print "Using forced SHA1 from env"

print "Will execute for", sha_to_run

for sha in sha_to_run:
    if not ensure_git_clone(sha):
        print "Error with git repo handling."
        sys_exit_safe()

    print "Ready for", sha

    print "Let us place ourselves into the git clone directory ..."
    root_dir = os.getcwd()
    os.chdir(DEEPSPEECH_CLONE_PATH)

    print "Copy previous run logs from backup"
    populate_previous_logs()

    print "Do the training for getting WER computation"
    exec_wer_run()

    print "Backup the logs we just produced"
    save_logs()

    print "Save progress"
    write_last_sha1(sha)

    print "Let us place back to the previous directory %s ..." % root_dir
    os.chdir(root_dir)

print "Getting rid of git clone"
wipe_git_clone()

release_lock()
