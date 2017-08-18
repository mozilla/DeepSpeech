Overview of the overall process for publishing WER
==================================================

The tracking of WER is made using the following workflow:
* a dedicated user on the learning machine periodically runs training jobs (cron
  job, or manual runs)
* this produces, mostly, js/hyper.js containing a concatenated version of all
  previous runs
* util/website.py contains code that will connect to an SSH server, using SFTP
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
*/15 *  *   *   *    test ! -f $HOME/.cache/deepspeech_wer/lock && (rm $HOME/.deepspeech_wer.locked.log; mkdir -p $HOME/wer && cd $HOME/wer && source /usr/local/tensorflow-env/bin/activate && /usr/bin/curl -H "Cache-Control: no-cache" -L https://raw.githubusercontent.com/mozilla/DeepSpeech/master/util/automation.py | ds_website_username="UUU" ds_website_privkey="FFF" ds_website_server_fqdn="SSS" ds_website_server_root="www/" ds_gpu_usage_root="/data/automation/gpu/" ds_dataroot="/data/" ds_wer_automation="./bin/run-wer-automation.sh" python -u ; cd) 2>$HOME/.deepspeech_wer.err.log | /usr/bin/ts "\%Y-\%m-\%d \%H:\%M:\%S" > $HOME/.deepspeech_wer.out.log || TZ='Europe/Berlin' date --rfc-2822 >> $HOME/.deepspeech_wer.locked.log
```
* Cron task will take care of:
 * checking if there were any new merges
 * perform a clone of the git repo and checkout those merges
 * schedule sequential execution against those merges
 * notebook is configured to automatically perform merging and to upload if
   the proper environment variables are configured, effectively updating the
   website on each iteration from the above process
 * saving of the hyper.json files produced
 * wiping the cloned git repo
* A 'lock' file will be created in ~/.cache/deepspeech_wer/ to ensure that we do not
  trigger multiple executions at the same time. Unexpected exception might leave
  a stale lock file
* A 'last_sha1' in the same directory will be used to keep track of what has
  been done last
* Previous runs' logs will be saved to ~/.local/share/deepspeech_wer/
* For debugging purpose, `~/.deepspeech_wer.err.log` and `~/.deepspeech_wer.out.log`
  will collect stderr/stdout
* Exposing the environment variables (please refer to util/website.py to have
  more details on each) (cron above does it):
 * ds_website_username
 * ds_website_privkey
 * ds_website_server_fqdn
 * ds_website_server_port
 * ds_website_server_root

# Setup of the web-facing server:

* Ensure existing webroot
* Generate an SSH key, and upload the public key to web-facing server
* Connect at least one time manually from the training machine to the web-facing
  server to accept the server host key and populate known_hosts file (pay
  attention to the FQDN)
* Make sure that server is configured with proper DirectoryIndex (Apache, or
  equivalent directive for others), whether system-wide or locally (with a
  .htaccess for example).
* Bootstrap with empty index.htm (and populate .htaccess if needed)
* That should be all. Upon any big changes with the HTML codebase, make sure to
  clean up the mess.
