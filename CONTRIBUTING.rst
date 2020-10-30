Contribution guidelines
=======================

Welcome to the DeepSpeech project! We are excited to see your interest, and appreciate your support!

This repository is governed by Mozilla's code of conduct and etiquette guidelines. For more details, please read the `Mozilla Community Participation Guidelines <https://www.mozilla.org/about/governance/policies/participation/>`_.

How to Make a Good Pull Request
-------------------------------

Here's some guidelines on how to make a good PR to DeepSpeech.

Bug-fix PR
^^^^^^^^^^

You've found a bug and you were able to squash it! Great job! Please write a short but clear commit message describing the bug, and how you fixed it. This makes review much easier. Also, please name your branch something related to the bug-fix.

Documentation PR
^^^^^^^^^^^^^^^^

If you're just making updates or changes to the documentation, there's no need to run all of DeepSpeech's tests for Continuous Integration (i.e. Taskcluster tests). In this case, at the end of your short but clear commit message, you should add **X-DeepSpeech: NOBUILD**. This will trigger the CI tests to skip your PR, saving both time and compute.

New Feature PR
^^^^^^^^^^^^^^

You've made some core changes to DeepSpeech, and you would like to share them back with the community -- great! First things first: if you're planning to add a feature (not just fix a bug or docs) let the DeepSpeech team know ahead of time and get some feedback early. A quick check-in with the team can save time during code-review, and also ensure that your new feature fits into the project.

The DeepSpeech codebase is made of many connected parts. There is Python code for training DeepSpeech, core C++ code for running inference on trained models, and multiple language bindings to the C++ core so you can use DeepSpeech in your favorite language.

Whenever you add a new feature to DeepSpeech and what to contribute that feature back to the project, here are some things to keep in mind:

1. You've made changes to the core C++ code. Core changes can have downstream effects on all parts of the DeepSpeech project, so keep that in mind. You should minimally also make necessary changes to the C client (i.e. **args.h** and **client.cc**). The bindings for Python, Java, and Javascript are SWIG generated, and in the best-case scenario you won't have to worry about them. However, if you've added a whole new feature, you may need to make custom tweaks to those bindings, because SWIG may not automagically work with your new feature, especially if you've exposed new arguments. The bindings for .NET and Swift are not generated automatically. It would be best if you also made the necessary manual changes to these bindings as well. It is best to communicate with the core DeepSpeech team and come to an understanding of where you will likely need to work with the bindings. They can't predict all the bugs you will run into, but they will have a good idea of how to plan for some obvious challenges.
2. You've made changes to the Python code. Make sure you run a linter (described below).
3. Make sure your new feature doesn't regress the project. If you've added a significant feature or amount of code, you want to be sure your new feature doesn't create performance issues. For example, if you've made a change to the DeepSpeech decoder, you should know that inference performance doesn't drop in terms of latency, accuracy, or memory usage. Unless you're proposing a new decoding algorithm, you probably don't have to worry about affecting accuracy. However, it's very possible you've affected latency or memory usage. You should run local performance tests to make sure no bugs have crept in. There are lots of tools to check latency and memory usage, and you should use what is most comfortable for you and gets the job done. If you're on Linux, you might find [[perf](https://perf.wiki.kernel.org/index.php/Main_Page)] to be a useful tool. You can use sample WAV files for testing which are provided in the `DeepSpeech/data/` directory.

Python Linter
-------------

Before making a Pull Request for Python code changes, check your changes for basic mistakes and style problems by using a linter. We have cardboardlinter setup in this repository, so for example, if you've made some changes and would like to run the linter on just the changed code, you can use the follow command:

.. code-block:: bash

   pip install pylint cardboardlint
   cardboardlinter --refspec master

This will compare the code against master and run the linter on all the changes. We plan to introduce more linter checks (e.g. for C++) in the future. To run it automatically as a git pre-commit hook, do the following:

.. code-block:: bash

   cat <<\EOF > .git/hooks/pre-commit
   #!/bin/bash
   if [ ! -x "$(command -v cardboardlinter)" ]; then
       exit 0
   fi

   # First, stash index and work dir, keeping only the
   # to-be-committed changes in the working directory.
   echo "Stashing working tree changes..." 1>&2
   old_stash=$(git rev-parse -q --verify refs/stash)
   git stash save -q --keep-index
   new_stash=$(git rev-parse -q --verify refs/stash)

   # If there were no changes (e.g., `--amend` or `--allow-empty`)
   # then nothing was stashed, and we should skip everything,
   # including the tests themselves.  (Presumably the tests passed
   # on the previous commit, so there is no need to re-run them.)
   if [ "$old_stash" = "$new_stash" ]; then
       echo "No changes, skipping lint." 1>&2
       exit 0
   fi

   # Run tests
   cardboardlinter --refspec HEAD -n auto
   status=$?

   # Restore changes
   echo "Restoring working tree changes..." 1>&2
   git reset --hard -q && git stash apply --index -q && git stash drop -q

   # Exit with status from test-run: nonzero prevents commit
   exit $status
   EOF
   chmod +x .git/hooks/pre-commit

This will run the linters on just the changes made in your commit.

