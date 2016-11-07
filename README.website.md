Overview of the process for publishing WER
==========================================

The tracking of WER is made using the following workflow:
* a dedicated user on the learning machine periodically runs training jobs (cron
  job, or manual runs)
* this produces, mostly, js/hyper.js containig a concatenated version of all
  previous runs
* util/website.py contains code that will connect to a SSH server, using SFTP
* this will publish 'index.html' and its dependencies

# Setup of the dedicated user:

* Create a standard user
* Either rely on system's tensorflow or populate a virtualenv
* Using system tensorflow or a virtualenv might require setting the PYTHONPATH
  env variable (done for system wide tensorflow installation in the example
  below).
* Install PIP dependencies:
 * jupyter
 * BeautifulSoup4
 * GitPython
 * pysftp
 * pyxdg
 * requests
* Construct cron job:
```
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin/:/bin
# Run WER every 15 mins
*/5 *  *   *   *    (mkdir -p $HOME/wer && cd $HOME/wer && source /usr/local/tensorflow-env/bin/activate && /usr/bin/curl -H "Cache-Control: no-cache" -L https://raw.githubusercontent.com/mozilla/DeepSpeech/website/util/automation.py | ds_website_username="u" ds_website_privkey="$HOME/.ssh/k" ds_website_server_fqdn="host.tld" ds_website_server_root="www/" ds_wer_automation="./bin/run-wer-automation.sh" python ; cd) 2>$HOME/.deepspeech_wer.err.log 1>$HOME/.deepspeech_wer.out.log
```
* Cron task will take care of:
 * checking if any there were any new merges
 * perform a clone of the git repo and checkout those merges
 * schedule sequential execution against those merges
 * notebook is configured to automatically perform merging and upload if
   the proper environment variables are configured, effectively updating the
   website on each iteration from the above process
 * saving of the hyper.json files produced
 * wiping the cloned git repo
* A 'lock' file will be created in ~/.cache/deepspeech_wer/ to ensure we do not
  trigger multiple execution at the same time. Unexpected exception might leave
  a stale lock file
* A 'last_sha1' in the same directory will be used to keep track of what has
  been done last
* Previous runs' logs will be saved to ~/.local/share/deepspeech_wer/
* For debugging purpose, `~/.deepspeech_wer.err.log` and `~/.deepspeech_wer.out.log`
  will collect stderr/stdout
* Expose those environment variable (please refer to util/website.py to have
  more details on each) (cron above does it):
 * ds_website_username
 * ds_website_privkey
 * ds_website_server_fqdn
 * ds_website_server_port
 * ds_website_server_root

# Setup of web-facing server:

* Ensure existing webroot
* Generate a SSH key, and upload public key to web-facing server
* Connect at least one time manually from the training machine to the web-facing
  server to accept the server host key and populate known_hosts file (pay
  attention to the FQDN)
* Make sure that server is configured with proper DirectoryIndex (Apache, or
  equivalent directive for others), whether system-wide or locally (with a
  .htaccess for example).
* Bootstrap with empty index.htm (and populate .htaccess if needed)
* That should be all. Upon any big changes with the HTML codebase, make sure to
  cleanup the mess.
