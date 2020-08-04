
Javadoc for Sphinx
==================

This code is only here for reference for documentation generation.

To update, please install SWIG (4.0 at least) and then run from native_client/java:

.. code-block::

   swig -c++ -java -doxygen -package org.mozilla.voice.stt -outdir libmozillavoicestt/src/main/java/org/mozilla/voice/stt_doc -o jni/deepspeech_wrap.cpp jni/deepspeech.i
