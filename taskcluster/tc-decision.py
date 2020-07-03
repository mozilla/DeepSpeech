# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from glob import glob
from functools import reduce

import json
import jsone
import os
import sys
import requests
import slugid
import yaml
import subprocess

import networkx as nx

TASKS_ROOT = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])))
TASKCLUSTER_API_BASEURL = 'http://taskcluster/queue/v1/task/%(task_id)s'

def string_to_dict(sid, value):
    parts = sid.split('.')

    def pack(parts):
        if len(parts) == 1:
            return {parts[0]: value}
        elif len(parts):
            return {parts[0]: pack(parts[1:])}
        return parts

    return pack(parts)

def merge_dicts(*dicts):
    if not reduce(lambda x, y: isinstance(y, dict) and x, dicts, True):
        raise TypeError("Object in *dicts not of type dict")
    if len(dicts) < 2:
        raise ValueError("Requires 2 or more dict objects")

    def merge(a, b):
        for d in set(a.keys()).union(b.keys()):
            if d in a and d in b:
                if type(a[d]) == type(b[d]):
                    if not isinstance(a[d], dict):
                        ret = list({a[d], b[d]})
                        if len(ret) == 1: ret = ret[0]
                        yield (d, sorted(ret))
                    else:
                        yield (d, dict(merge(a[d], b[d])))
                else:
                    raise TypeError("Conflicting key:value type assignment", type(a[d]), a[d], type(b[d]), b[d])
            elif d in a:
                yield (d, a[d])
            elif d in b:
                yield (d, b[d])
            else:
                raise KeyError

    return reduce(lambda x, y: dict(merge(x, y)), dicts[1:], dicts[0])

def taskcluster_event_context():
    das_context = {}

    # Pre-filterting
    for k in os.environ.keys():
        if k == 'GITHUB_HEAD_USER':
            os.environ['GITHUB_HEAD_USER_LOGIN'] = os.environ[k]
            del os.environ['GITHUB_HEAD_USER']

    for k in os.environ.keys():
        if k == 'TASK_ID':
            parts = string_to_dict('taskcluster.taskGroupId', os.environ[k])
            das_context = merge_dicts(das_context, parts)

        if k.startswith('GITHUB_'):
            parts = string_to_dict(k.lower().replace('_', '.').replace('github', 'event'), os.environ[k])
            das_context = merge_dicts(das_context, parts)

    return das_context

def load_specific_contextFile(file):
    specific_context = {}

    try:
        with open(os.path.join(TASKS_ROOT, file)) as src:
            specific_context = yaml.load(src)

        if specific_context is None:
            specific_context = {}
    except FileNotFoundError:
        specific_context = {}

    return specific_context

def defaultValues_build_context():
    return load_specific_contextFile('.build.yml')

def shared_context():
    return load_specific_contextFile('.shared.yml')

def create_task_payload(build, base_context):
    print('build', build)
    build_type = os.path.splitext(os.path.basename(build))[0]

    build_context = defaultValues_build_context()
    with open(build) as src:
        build_context['build'].update(yaml.load(src)['build'])

    # Be able to use what has been defined in base_context
    # e.g., the {${event.head.branch}}
    build_context = jsone.render(build_context, base_context)
    template_context = {
        'taskcluster': {
            'taskId': as_slugid(build_type)
        },
        'build_type': build_type
    }

    with open(os.path.join(TASKS_ROOT, build_context['build']['template_file'])) as src:
        template = yaml.load(src)

    contextes = merge_dicts({}, base_context, template_context, build_context)
    for one_context in glob(os.path.join(TASKS_ROOT, '*.cyml')):
        with open(one_context) as src:
            contextes = merge_dicts(contextes, yaml.load(src))

    return jsone.render(template, contextes)

def send_task(t):
    url = TASKCLUSTER_API_BASEURL % {'task_id': t['taskId']}
    del t['taskId']

    r = requests.put(url, json=t)

    print(url, r.status_code)
    if r.status_code != requests.codes.ok:
        print(json.dumps(t, indent=2))
        print(r.content)
        print(json.loads(r.content.decode())['message'])

    return r.status_code == requests.codes.ok

slugids = {}
def as_slugid(name):
    if name not in slugids:
        slugids[name] = slugid.nice().decode()
        print('cache miss', name, slugids[name])
    else:
        print('cache hit', name, slugids[name])
    return slugids[name]

def to_int(x):
    return int(x)

def functions_context():
    return {
        'as_slugid': as_slugid,
        'to_int': to_int
    }

def is_dry_run():
    return (len(sys.argv) > 1) and (sys.argv[1] == '--dry')

def should_run():
    # Make a quick clone to fetch the last commit
    try:
        subprocess.check_call([
            'git', 'clone', '--quiet', '-b', os.environ.get('GITHUB_HEAD_BRANCH'),
            '--single-branch', os.environ.get('GITHUB_HEAD_REPO_URL'),
            '--depth=1', '/tmp/ds-clone/'
        ], env={'GIT_LFS_SKIP_SMUDGE': '1'})
    except subprocess.CalledProcessError as e:
        print("Error while git cloning:", e, file=sys.stderr)
        return False

    try:
        git_msg = subprocess.check_output([
            'git', '--git-dir=/tmp/ds-clone/.git/',
            'log', '--format=%b', '-n', '1',
            os.environ.get('GITHUB_HEAD_SHA')
        ]).decode('utf-8').strip().upper()
    except subprocess.CalledProcessError as e:
        print("Error while git show:", e, file=sys.stderr)
        return False

    print('Commit message:', git_msg)

    x_deepspeech = filter(lambda x: 'X-DEEPSPEECH:' in x, git_msg.split('\n'))
    if len(list(filter(lambda x: 'NOBUILD' in x, x_deepspeech))) == 1:
        print('Not running anything according to commit message')
        return False

    return True

if __name__ == '__main__':
    if not is_dry_run():
        # We might want to NOT run in some cases
        if not should_run():
            sys.exit(0)

    base_context = taskcluster_event_context()
    base_context = merge_dicts(base_context, functions_context())
    base_context = merge_dicts(base_context, shared_context())

    root_task = base_context['taskcluster']['taskGroupId']

    tasks_graph = nx.DiGraph()
    tasks = {}

    for build in glob(os.path.join(TASKS_ROOT, '*.yml')):
        t = create_task_payload(build, base_context)

        # We allow template to produce completely empty output
        if not t:
            continue

        if 'dependencies' in t and len(t['dependencies']) > 0:
            for dep in t['dependencies']:
                tasks_graph.add_edge(t['taskId'], dep)
        else:
            tasks_graph.add_edge(t['taskId'], root_task)

        tasks[t['taskId']] = t

    for task in nx.dfs_postorder_nodes(tasks_graph):
        # root_task is the task group and also the task id that is already
        # running, so we don't have to schedule that
        if task == root_task:
            continue

        t = tasks[task]
        if is_dry_run():
            print(json.dumps(t, indent=2))
            continue

        p = send_task(t)
        if not p:
            sys.exit(1)
