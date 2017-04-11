from __future__ import print_function
from __future__ import absolute_import
import os
import paramiko
import pysftp
import sys

from bs4 import BeautifulSoup

def parse_for_deps(filename):
    """
    This takes an HTML file as input and output a list of existing depenencies.
    Empty output means something was wrong.
    Dependency is one of: js script loaded, css stylesheet loaded
    """

    with open(filename, 'r') as code:
        soup = BeautifulSoup(code.read(), 'html.parser')

    external_links   = filter(lambda x: x is not None, [ link.get('href') for link in soup.find_all('link') ])
    external_scripts = filter(lambda x: x is not None, [ script.get('src') for script in soup.find_all('script') ])

    # Verify with stat()
    try:
        all_resources = map(lambda x: os.stat(x), external_links + external_scripts)
    except OSError as ex:
        if ex.errno == 2:
            print("Missing dependency", ex)
            return []
        raise ex

    # Try to read the file
    try:
        readable_resources = map(lambda x: os.open(x, os.O_RDONLY), external_links + external_scripts)
        map(lambda x: os.close(x), readable_resources)
    except OSError as ex:
        print("Unreadable dependency", ex)
        return []

    # It should be all good.
    return external_links + external_scripts

def get_ssh(auth_infos):
    """
    Connect to SSH server
    """
    cinfo = {'host':        auth_infos['ds_website_server_fqdn'],
             'username':    auth_infos['ds_website_username'],
             'private_key': auth_infos['ds_website_privkey'],
             'port':        auth_infos['ds_website_server_port']}
    return pysftp.Connection(**cinfo)

def verify_ssh_dir(auth_infos):
    """
    This should ensure that SSH connection works and that the directory where
    we want to push data is either empty or contains at least a .htaccess and
    index.htm file
    """
    print("Checking connection: %s@%s:%s (port %d)" % (auth_infos['ds_website_username'], auth_infos['ds_website_server_fqdn'], auth_infos['ds_website_server_root'], auth_infos['ds_website_server_port']))

    try:
        with get_ssh(auth_infos) as sftp:
            with sftp.cd(auth_infos['ds_website_server_root']):
                files = sftp.listdir()
                if len(files) == 0 or (('.htaccess' in files) and ('index.htm' in files)):
                    return True
                else:
                    print("Invalid content for %s:" % (auth_infos['ds_website_server_root']), files)
                    return False
    except paramiko.ssh_exception.AuthenticationException as ex:
        print("Authentication error, please verify credentials")
        return False
    except IOError as ex:
        print("Unable to read a file (private key or invalid path on server ?)", ex)
        return False

    # Should not be reached
    return False

def push_files_sftp(files, auth_infos):
    """
    This will push all of the ``files`` listed to the server root folder.
    """

    # Directories that we might be required to create
    dirs = filter(lambda x: len(x) > 0, list(set([ os.path.dirname(x) for x in files ])))

    created = []
    pushed = []

    try:
        with get_ssh(auth_infos) as sftp:
            with sftp.cd(auth_infos['ds_website_server_root']):
                # Create dirs if needed, they all should depend from the root
                for dir in dirs:
                    if not sftp.isdir(dir):
                        print("Creating directory", dir)
                        sftp.makedirs(dir)
                        created.append(dir)

                # Push all files, chdir() for each
                for fil in files:
                    with sftp.cd(os.path.dirname(fil)):
                        print("Pushing", fil)
                        sftp.put(fil)
                        pushed.append(fil)

        return pushed
    except paramiko.ssh_exception.AuthenticationException as ex:
        print("Authentication error, please verify credentials")
        return False
    except IOError as ex:
        print("Unable to read a file (private key or invalid path on server ?)", ex)
        return False

    # Should not be reached
    return False

def maybe_publish(file='index.htm'):
    """
    Publishing to the web requires the following env variables to be set
        'ds_website_username: defines the SSH username to connect to
        'ds_website_privkey: defines the SSH privkey filename to use for connection
        'ds_website_server_fqdn: hostname of the SSH server
        'ds_website_server_port: port of the SSH server, defaults to 22
        'ds_website_server_root: directory on the server where to push data,
                               should be either empty, or containing at least
                               a .htaccess and index.htm file
    """

    ssh_auth_infos = {
        'ds_website_username': '',
        'ds_website_privkey': '',
        'ds_website_server_fqdn': '',
        'ds_website_server_port': 22,
        'ds_website_server_root': ''
    }

    for key in ssh_auth_infos.keys():
        vartype = type(ssh_auth_infos[key])
        value = os.environ.get(key)
        if value is not None:
            if vartype == str:
                ssh_auth_infos[key] = str(os.environ.get(key))
            elif vartype == int:
                try:
                    ssh_auth_infos[key] = int(os.environ.get(key))
                except TypeError as ex:
                    print("WARNING:", "Keeping default SSH port value because of:", ex)
    missing_env = [
        x for x in ssh_auth_infos.keys()
        if (len(str(ssh_auth_infos[x])) == 0)
    ]
    if len(missing_env) > 0:
        print("Not publishing, missing some required environment variables:", missing_env)
        print("But maybe this is what you wanted, after all ...")
        return False

    all_deps = parse_for_deps(file)

    if len(all_deps) == 0:
        print("Problem during deps computation, aborting")
        return False

    if not verify_ssh_dir(ssh_auth_infos):
        print("Problem during SSH directory verification, aborting")
        return False

    all_files = [ file ] + all_deps
    uploaded = push_files_sftp(all_files, ssh_auth_infos)
    if len(uploaded) == 0:
        print("Unable to upload anything")
        return False
    elif len(uploaded) != len(all_files):
        print("Partial upload has been completed:")
        print("all_files=", all_files)
        print("uploaded=", uploaded)
    else:
        print("Complete upload has been completed.")

    return True

# Support CLI invocation
if __name__ == "__main__":
    if maybe_publish():
        print("All good!")
        sys.exit(0)
    else:
        print("Error happened ...")
        sys.exit(1)
