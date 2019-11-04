Utility scripts
===============

This folder contains scripts that can be used to do training on the various included importers from the command line. This is useful to be able to run training without a browser open, or unattended on a remote machine. They should be run from the base directory of the repository. Note that the default settings assume a very well-specified machine. In the situation that out-of-memory errors occur, you may find decreasing the values of ``--train_batch_size``\ , ``--dev_batch_size`` and ``--test_batch_size`` will allow you to continue, at the expense of speed.
