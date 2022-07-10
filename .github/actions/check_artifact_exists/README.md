Building and using a TensorFlow cache:
======================================

The present action will check the existence of an artifact in the list of the
repo artifacts. Since we don't want always to download the artifact, we can't
rely on the official download-artifact action.

Rationale:
----------

Because of the amount of code required to build TensorFlow, the library build
is split into two main parts to make it much faster to run PRs:
 - a TensorFlow prebuild cache
 - actual code of the library

The TensorFlow prebuild cache exists because building tensorflow (even just the
`libtensorflow_cpp.so`) is a huge amount of code and it will take several hours
even on decent systems. So we perform a cache build of it, because the
tensorflow version does not change that often.

However, each PR might have changes to the actual library code, so we rebuild
this everytime.

The `tensorflow_opt-macOS` job checks whether such build cache exists alrady.
Those cache are stored as artifacts because [GitHub Actions
cache](https://docs.github.com/en/actions/guides/caching-dependencies-to-speed-up-workflows)
has size limitations.

The `build-tensorflow-macOS` job has a dependency against the cache check to
know whether it needs to run an actual build or not.

Hacking:
--------

For hacking into the action, please follow the [GitHub JavaScript
Actions](https://docs.github.com/en/actions/creating-actions/creating-a-javascript-action#commit-tag-and-push-your-action-to-github)
and specifically the usage of `ncc`.

```
$ npm install
$ npx ncc build main.js --license licenses.txt
$ git add dist/
```
