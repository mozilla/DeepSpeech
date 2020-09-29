#!/bin/bash

set -xe

tc_tests_utils=$(dirname ${BASH_SOURCE[0]})

# This defines a set of variables used by others
source ${tc_tests_utils}/tc-all-vars.sh

# Useful functions for setup / generic re-use like downloading data
source ${tc_tests_utils}/tc-all-utils.sh

# Scoping of Android-related tooling
source ${tc_tests_utils}/tc-android-utils.sh

# Scoping of Python-related tooling
source ${tc_tests_utils}/tc-py-utils.sh

# Scoping of Node-related tooling
source ${tc_tests_utils}/tc-node-utils.sh

# Scoping of .Net-related tooling
source ${tc_tests_utils}/tc-dotnet-utils.sh

# For checking with valgrind
source ${tc_tests_utils}/tc-valgrind-utils.sh

# Functions that controls directly the build process
source ${tc_tests_utils}/tc-build-utils.sh

# Functions that allows to assert model behavior
source ${tc_tests_utils}/tc-asserts.sh

# Handling TaskCluster artifacts / packaging
source ${tc_tests_utils}/tc-package.sh
