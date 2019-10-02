
Making a (new) release of the codebase
======================================


* Update version in VERSION file, commit
* Open PR, ensure all tests are passing properly
* Merge the PR
* Fetch the new master, tag it with (hopefully) the same version as in VERSION
* Push that to Github
* New build should be triggered and new packages should be made
* TaskCluster should schedule a merge build **including** a "DeepSpeech Packages" task
